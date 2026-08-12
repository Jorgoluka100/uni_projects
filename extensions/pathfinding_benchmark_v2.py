"""Reproducible pathfinding benchmark suite for the restored Pathfinding notebook.

Adds evidence the original algorithm demo was missing: seeded grid families, BFS,
Dijkstra and A* under a shared contract, path-cost verification, optimality checks,
expanded-node counts, runtime summaries and an admissibility sanity test.

Usage:
    python extensions/pathfinding_benchmark_v2.py --output-dir pathfinding_artifacts
    python extensions/pathfinding_benchmark_v2.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Point = tuple[int, int]


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    cost: float
    path: list[Point]
    expanded: int
    runtime_ms: float


def neighbours(point: Point, grid: list[list[int]]) -> list[Point]:
    row, col = point
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0]


def reconstruct(parent: dict[Point, Point | None], goal: Point) -> list[Point]:
    path: list[Point] = []
    current: Point | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def bfs(grid: list[list[int]], start: Point, goal: Point) -> SearchResult:
    begun = time.perf_counter()
    queue = deque([start]); parent: dict[Point, Point | None] = {start: None}; expanded = 0
    while queue:
        node = queue.popleft(); expanded += 1
        if node == goal:
            path = reconstruct(parent, goal)
            return SearchResult("BFS", True, float(len(path) - 1), path, expanded, (time.perf_counter() - begun) * 1000)
        for nxt in neighbours(node, grid):
            if nxt not in parent:
                parent[nxt] = node; queue.append(nxt)
    return SearchResult("BFS", False, math.inf, [], expanded, (time.perf_counter() - begun) * 1000)


def best_first(grid: list[list[int]], start: Point, goal: Point, heuristic: Callable[[Point, Point], float], name: str) -> SearchResult:
    begun = time.perf_counter(); frontier: list[tuple[float, float, Point]] = [(heuristic(start, goal), 0.0, start)]
    parent: dict[Point, Point | None] = {start: None}; best = {start: 0.0}; expanded = 0
    while frontier:
        _, cost, node = heapq.heappop(frontier)
        if cost != best.get(node):
            continue
        expanded += 1
        if node == goal:
            return SearchResult(name, True, cost, reconstruct(parent, goal), expanded, (time.perf_counter() - begun) * 1000)
        for nxt in neighbours(node, grid):
            candidate = cost + 1.0
            if candidate < best.get(nxt, math.inf):
                best[nxt] = candidate; parent[nxt] = node
                heapq.heappush(frontier, (candidate + heuristic(nxt, goal), candidate, nxt))
    return SearchResult(name, False, math.inf, [], expanded, (time.perf_counter() - begun) * 1000)


def dijkstra(grid: list[list[int]], start: Point, goal: Point) -> SearchResult:
    return best_first(grid, start, goal, lambda _a, _b: 0.0, "Dijkstra")


def manhattan(a: Point, b: Point) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def astar(grid: list[list[int]], start: Point, goal: Point) -> SearchResult:
    return best_first(grid, start, goal, manhattan, "A*")


def make_grid(size: int, obstacle_rate: float, seed: int) -> tuple[list[list[int]], Point, Point]:
    rng = random.Random(seed); start = (0, 0); goal = (size - 1, size - 1)
    for _ in range(100):
        grid = [[1 if rng.random() < obstacle_rate else 0 for _ in range(size)] for _ in range(size)]
        grid[start[0]][start[1]] = 0; grid[goal[0]][goal[1]] = 0
        if bfs(grid, start, goal).found:
            return grid, start, goal
    raise RuntimeError("could not generate a solvable seeded grid")


def validate_path(grid: list[list[int]], path: list[Point], start: Point, goal: Point) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False
    for left, right in zip(path, path[1:]):
        if right not in neighbours(left, grid):
            return False
    return True


def run_suite() -> list[dict[str, object]]:
    cases = [(25, 0.10, 11), (25, 0.20, 22), (40, 0.20, 33), (40, 0.28, 44), (60, 0.22, 55)]
    rows: list[dict[str, object]] = []
    for size, rate, seed in cases:
        grid, start, goal = make_grid(size, rate, seed)
        results = [bfs(grid, start, goal), dijkstra(grid, start, goal), astar(grid, start, goal)]
        optimal = results[1].cost
        for result in results:
            if result.found and not validate_path(grid, result.path, start, goal):
                raise AssertionError(f"invalid path from {result.algorithm}")
            rows.append({
                "size": size, "obstacle_rate": rate, "seed": seed,
                "algorithm": result.algorithm, "found": result.found, "cost": result.cost,
                "optimal_cost": optimal, "optimal": result.cost == optimal,
                "expanded": result.expanded, "runtime_ms": result.runtime_ms,
            })
    return rows


def self_test() -> None:
    grid = [[0,0,0],[1,1,0],[0,0,0]]; start=(0,0); goal=(2,2)
    results = [bfs(grid,start,goal), dijkstra(grid,start,goal), astar(grid,start,goal)]
    assert all(item.found for item in results)
    assert {item.cost for item in results} == {4.0}
    assert all(validate_path(grid,item.path,start,goal) for item in results)
    # Manhattan cannot overestimate the true shortest path on a 4-neighbour unit grid.
    assert manhattan(start, goal) <= dijkstra(grid,start,goal).cost
    rows = run_suite()
    assert all(bool(row["optimal"]) for row in rows)
    print("Pathfinding benchmark self-test passed.")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=Path("pathfinding_artifacts")); parser.add_argument("--self-test", action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test(); return 0
    rows=run_suite(); args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path=args.output_dir/"benchmark.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={}
    for algo in ("BFS","Dijkstra","A*"):
        subset=[row for row in rows if row["algorithm"]==algo]
        summary[algo]={
            "all_optimal": all(bool(row["optimal"]) for row in subset),
            "median_expanded": statistics.median(float(row["expanded"]) for row in subset),
            "median_runtime_ms": statistics.median(float(row["runtime_ms"]) for row in subset),
        }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
