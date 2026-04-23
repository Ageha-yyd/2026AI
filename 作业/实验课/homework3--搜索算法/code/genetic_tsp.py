from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

EPSILON = 1e-12


class GeneticAlgTSP:
    def __init__(
        self,
        filename: str,
        population_size: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        elite_size: int = 2,
        tournament_size: int = 4,
        random_seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if elite_size < 0 or elite_size >= population_size:
            raise ValueError("elite_size must be in [0, population_size)")
        if tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        if tournament_size > population_size:
            raise ValueError("tournament_size must be <= population_size")

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.cities = self._read_tsp(filename)
        self.num_cities = self.cities.shape[0]
        if self.num_cities < 2:
            raise ValueError("TSP must contain at least 2 cities")

        self.distance_matrix = self._build_distance_matrix(self.cities)
        self.population = self._init_population()

    def _read_tsp(self, filename: str) -> np.ndarray:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"TSP file not found: {filename}")

        coords: List[Tuple[float, float]] = []
        in_coord = False
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                upper = line.upper()
                if upper.startswith("NODE_COORD_SECTION"):
                    in_coord = True
                    continue
                if upper.startswith("EOF"):
                    break
                if not in_coord:
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))

        if not coords:
            raise ValueError("No city coordinates were parsed from TSP file")
        return np.array(coords, dtype=np.float64)

    def _build_distance_matrix(self, cities: np.ndarray) -> np.ndarray:
        n = cities.shape[0]
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.dist(cities[i], cities[j])
                matrix[i, j] = d
                matrix[j, i] = d
        return matrix

    def _init_population(self) -> List[np.ndarray]:
        base = np.arange(self.num_cities, dtype=np.int32)
        population: List[np.ndarray] = []
        for _ in range(self.population_size):
            chromosome = base.copy()
            np.random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _tour_length(self, chromosome: np.ndarray) -> float:
        d = 0.0
        for i in range(self.num_cities):
            a = chromosome[i]
            b = chromosome[(i + 1) % self.num_cities]
            d += self.distance_matrix[a, b]
        return d

    def _fitness(self, chromosome: np.ndarray) -> float:
        return 1.0 / (self._tour_length(chromosome) + EPSILON)

    def _select_parent(self, fitness_values: Sequence[float]) -> np.ndarray:
        idxs = np.random.choice(len(self.population), size=self.tournament_size, replace=False)
        best_idx = max(idxs, key=lambda i: fitness_values[i])
        return self.population[int(best_idx)]

    def _pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        n = self.num_cities
        child = np.full(n, -1, dtype=np.int32)
        s, t = sorted(np.random.choice(n, size=2, replace=False))
        child[s : t + 1] = parent1[s : t + 1]
        segment = set(int(g) for g in child[s : t + 1])

        for i in range(s, t + 1):
            gene = int(parent2[i])
            if gene in segment:
                continue
            pos = i
            # PMX mapping chain always ends because there is at least one
            # unfilled position outside the exchanged segment for this gene.
            while True:
                mapped_gene = int(parent1[pos])
                pos = int(np.where(parent2 == mapped_gene)[0][0])
                if child[pos] == -1:
                    child[pos] = gene
                    break

        for i in range(n):
            if child[i] == -1:
                child[i] = parent2[i]

        return child

    def _inversion_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        n = self.num_cities
        s, t = sorted(np.random.choice(n, size=2, replace=False))
        mutated = chromosome.copy()
        mutated[s : t + 1] = mutated[s : t + 1][::-1]
        return mutated

    def _best_individual(self) -> np.ndarray:
        lengths = [self._tour_length(ch) for ch in self.population]
        return self.population[int(np.argmin(lengths))]

    def iterate(self, num_iterations: int) -> List[int]:
        if num_iterations < 0:
            raise ValueError("num_iterations must be >= 0")

        for _ in range(num_iterations):
            lengths = [self._tour_length(ch) for ch in self.population]
            fitness_values = [1.0 / (length + EPSILON) for length in lengths]
            elite_indices = np.argsort(lengths)[: self.elite_size]
            next_population = [self.population[int(i)].copy() for i in elite_indices]

            while len(next_population) < self.population_size:
                p1 = self._select_parent(fitness_values)
                p2 = self._select_parent(fitness_values)

                if np.random.random() < self.crossover_rate:
                    child = self._pmx_crossover(p1, p2)
                else:
                    child = p1.copy()

                if np.random.random() < self.mutation_rate:
                    child = self._inversion_mutation(child)

                next_population.append(child)

            self.population = next_population

        best = self._best_individual()
        return [int(x) + 1 for x in best.tolist()]


if __name__ == "__main__":
    print("Usage example:")
    print("solver = GeneticAlgTSP('dj38.tsp', random_seed=42)")
    print("path = solver.iterate(200)")
