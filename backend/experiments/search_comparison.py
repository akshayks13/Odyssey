"""A* against uniform-cost search and greedy, on every subset of the region's cities.

    cd backend
    python -m experiments.search_comparison

For each number of cities it runs A* with the MST heuristic, A* with the weaker "cheapest edge out" heuristic, uniform-cost
search (h = 0) and greedy nearest neighbour on the road times UCS finds between towns. All three A*/UCS variants are
optimal (checked here against each other, and against brute force in the tests); what differs is how many nodes they
expand, and how often greedy misses the optimum.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.astar import astar_order, greedy_order  # noqa: E402
from experiments.run_experiments import RESULTS_DIR, _pyplot, write_csv  # noqa: E402
from tools import world  # noqa: E402

REGION = "Kerala, India"


def compare(region: str = REGION) -> list[dict]:
    cities = [c["name"] for c in world.cities(region)]
    dist = {a: {b: world.road_route(region, a, b)["hours"] for b in cities if b != a} for a in cities}
    rows = []
    for size in range(2, len(cities) + 1):
        results = []
        for subset in itertools.combinations(cities, size):
            sub = {a: {b: dist[a][b] for b in subset if b != a} for a in subset}
            names = list(subset)
            results.append((astar_order(names, sub, "mst"), astar_order(names, sub, "min_edge"), astar_order(names, sub, "zero"), greedy_order(names, sub)))
        n = len(results)
        rows.append({
            "cities": size,
            "subsets": n,
            "astar_mst_nodes": round(sum(r[0]["nodes_expanded"] for r in results) / n, 1),
            "astar_min_edge_nodes": round(sum(r[1]["nodes_expanded"] for r in results) / n, 1),
            "ucs_nodes": round(sum(r[2]["nodes_expanded"] for r in results) / n, 1),
            "all_optimal": all(abs(m["total_hours"] - u["total_hours"]) < 1e-6 and abs(e["total_hours"] - u["total_hours"]) < 1e-6 for m, e, u, _ in results),
            "greedy_optimal": sum(1 for m, _, _, g in results if g["total_hours"] <= m["total_hours"] + 1e-6),
            "greedy_extra_hours": round(sum(g["total_hours"] - m["total_hours"] for m, _, _, g in results) / n, 3),
        })
    return rows


def plot(out: Path, rows: list[dict]) -> None:
    plt = _pyplot()
    if plt is None:
        return
    sizes = [r["cities"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for key, label, color in (("ucs_nodes", "Uniform-cost search", "#9a9a9a"), ("astar_min_edge_nodes", "A* (min-edge heuristic)", "#d98a3d"), ("astar_mst_nodes", "A* (MST heuristic)", "#1e5c55")):
        axes[0].plot(sizes, [r[key] for r in rows], "o-", color=color, label=label)
    axes[0].set_yscale("log"); axes[0].set_xlabel("cities to order"); axes[0].set_ylabel("nodes expanded (mean)")
    axes[0].set_title("Same optimal order, far fewer nodes")
    axes[0].legend(frameon=False)
    axes[1].bar(sizes, [100 * r["greedy_optimal"] / r["subsets"] for r in rows], color="#b85c38")
    axes[1].set_xlabel("cities to order"); axes[1].set_ylabel("% of subsets where greedy is optimal"); axes[1].set_ylim(0, 105)
    axes[1].set_title("Nearest neighbour stops being optimal")
    fig.tight_layout(); fig.savefig(out / "search_comparison.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    rows = compare()
    write_csv(args.out / "search_comparison.csv", rows)
    print("| cities | subsets | A* MST nodes | A* min-edge nodes | UCS nodes | A*/UCS optimal | greedy optimal | greedy extra hours |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cities']} | {r['subsets']} | {r['astar_mst_nodes']} | {r['astar_min_edge_nodes']} | {r['ucs_nodes']} | {'yes' if r['all_optimal'] else 'NO'} | {r['greedy_optimal']}/{r['subsets']} | {r['greedy_extra_hours']:.2f} |")
    plot(args.out, rows)
    print(f"\nWrote search_comparison.csv and search_comparison.png to {args.out}")


if __name__ == "__main__":
    main()
