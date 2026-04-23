from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Sequence, Tuple

Board = Tuple[Tuple[int, ...], ...]
GOAL: Board = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 0))
GOAL_POS = {value: (r, c) for r, row in enumerate(GOAL) for c, value in enumerate(row)}
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Node:
    state: Board
    blank: Tuple[int, int]


def _to_board(puzzle: Sequence[Sequence[int]]) -> Board:
    if len(puzzle) != 4 or any(len(row) != 4 for row in puzzle):
        raise ValueError("15-Puzzle must be a 4x4 nested list.")
    values = [v for row in puzzle for v in row]
    if sorted(values) != list(range(16)):
        raise ValueError("Puzzle values must be numbers 0~15 without duplication.")
    return tuple(tuple(int(v) for v in row) for row in puzzle)


def _find_blank(state: Board) -> Tuple[int, int]:
    for r in range(4):
        for c in range(4):
            if state[r][c] == 0:
                return r, c
    raise ValueError("Puzzle has no blank tile (0).")


def _manhattan(state: Board) -> int:
    distance = 0
    for r in range(4):
        for c in range(4):
            tile = state[r][c]
            if tile == 0:
                continue
            gr, gc = GOAL_POS[tile]
            distance += abs(r - gr) + abs(c - gc)
    return distance


def _is_solvable(state: Board) -> bool:
    flat = [value for row in state for value in row if value != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            inversions += flat[i] > flat[j]
    blank_r, _ = _find_blank(state)
    row_from_bottom = 4 - blank_r
    return (inversions + row_from_bottom) % 2 == 1


def _swap(state: Board, a: Tuple[int, int], b: Tuple[int, int]) -> Board:
    ar, ac = a
    br, bc = b
    grid = [list(row) for row in state]
    grid[ar][ac], grid[br][bc] = grid[br][bc], grid[ar][ac]
    return tuple(tuple(row) for row in grid)


def _neighbors(node: Node):
    br, bc = node.blank
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            moved_tile = node.state[nr][nc]
            nxt_state = _swap(node.state, (br, bc), (nr, nc))
            yield Node(nxt_state, (nr, nc)), moved_tile


def A_star(puzzle: Sequence[Sequence[int]]) -> List[int]:
    start_state = _to_board(puzzle)
    if start_state == GOAL:
        return []
    if not _is_solvable(start_state):
        return []

    start = Node(start_state, _find_blank(start_state))
    open_heap: List[Tuple[int, int, Node]] = []
    heappush(open_heap, (_manhattan(start.state), 0, start))

    g_score: Dict[Node, int] = {start: 0}
    parent: Dict[Node, Tuple[Node, int]] = {}

    while open_heap:
        _, g, current = heappop(open_heap)
        if g != g_score.get(current, float("inf")):
            continue

        if current.state == GOAL:
            moves: List[int] = []
            trace = current
            while trace in parent:
                trace, moved_tile = parent[trace]
                moves.append(moved_tile)
            moves.reverse()
            return moves

        ng = g + 1
        for nxt, moved_tile in _neighbors(current):
            if ng < g_score.get(nxt, float("inf")):
                g_score[nxt] = ng
                parent[nxt] = (current, moved_tile)
                heappush(open_heap, (ng + _manhattan(nxt.state), ng, nxt))

    return []


def IDA_star(puzzle: Sequence[Sequence[int]]) -> List[int]:
    start_state = _to_board(puzzle)
    if start_state == GOAL:
        return []
    if not _is_solvable(start_state):
        return []

    start = Node(start_state, _find_blank(start_state))
    threshold = _manhattan(start.state)

    path_nodes = [start]
    path_moves: List[int] = []
    in_path = {start}

    def dfs(g: int, bound: int) -> Tuple[int, bool]:
        current = path_nodes[-1]
        f = g + _manhattan(current.state)
        if f > bound:
            return f, False
        if current.state == GOAL:
            return f, True

        min_bound = float("inf")
        for nxt, moved_tile in _neighbors(current):
            if nxt in in_path:
                continue
            in_path.add(nxt)
            path_nodes.append(nxt)
            path_moves.append(moved_tile)

            nxt_bound, found = dfs(g + 1, bound)
            if found:
                return nxt_bound, True
            min_bound = min(min_bound, nxt_bound)

            path_moves.pop()
            path_nodes.pop()
            in_path.remove(nxt)

        return min_bound, False

    while True:
        nxt_bound, found = dfs(0, threshold)
        if found:
            return path_moves.copy()
        if nxt_bound == float("inf"):
            return []
        threshold = int(nxt_bound)


if __name__ == "__main__":
    demo = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    print("A*:", A_star(demo))
    print("IDA*:", IDA_star(demo))
