"""Compare the visit-order searches on every subset of the region's cities (for the Review 2 report).

    cd backend && python -m scripts.compare_search

For each number of cities it runs A* with the MST heuristic, A* with the weaker min-edge heuristic,
uniform-cost search (h = 0) and greedy nearest neighbour, on road times from UCS, and prints the average
nodes expanded and how often greedy misses the optimum. Everything is deterministic.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.astar import astar_order, greedy_order  # noqa: E402
from tools import world  # noqa: E402

REGION = "Kerala, India"


def main() -> None:
    cities = [c["name"] for c in world.cities(REGION)]
    dist = {a: {b: world.road_route(REGION, a, b)["hours"] for b in cities if b != a} for a in cities}
    print("| cities | subsets | A* MST nodes | A* min-edge nodes | UCS nodes | all A*/UCS optimal | greedy optimal | greedy extra hours (avg) |")
    print("|---|---|---|---|---|---|---|---|")
    for size in range(2, len(cities) + 1):
        rows = []
        for subset in itertools.combinations(cities, size):
            sub = {a: {b: dist[a][b] for b in subset if b != a} for a in subset}
            mst, edge, ucs, greedy = (astar_order(list(subset), sub, "mst"), astar_order(list(subset), sub, "min_edge"),
                                      astar_order(list(subset), sub, "zero"), greedy_order(list(subset), sub))
            rows.append((mst, edge, ucs, greedy))
        n = len(rows)
        same = all(abs(m["total_hours"] - u["total_hours"]) < 1e-6 and abs(e["total_hours"] - u["total_hours"]) < 1e-6 for m, e, u, _ in rows)
        greedy_ok = sum(1 for m, _, _, g in rows if g["total_hours"] <= m["total_hours"] + 1e-6)
        extra = sum(g["total_hours"] - m["total_hours"] for m, _, _, g in rows) / n
        print(f"| {size} | {n} | {sum(r[0]['nodes_expanded'] for r in rows) / n:.1f} | {sum(r[1]['nodes_expanded'] for r in rows) / n:.1f} | "
              f"{sum(r[2]['nodes_expanded'] for r in rows) / n:.1f} | {'yes' if same else 'NO'} | {greedy_ok}/{n} | {extra:.2f} |")


if __name__ == "__main__":
    main()
