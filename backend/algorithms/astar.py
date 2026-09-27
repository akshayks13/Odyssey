"""A* search over the order in which to visit the chosen cities.

Problem formulation (an open travelling-salesman path):
    state      = (current city or None before the first stop, frozenset of cities visited)
    initial    = (None, {})
    actions    = travel to any city not yet visited
    step cost  = travel hours between the two cities (from UCS over the road/rail network);
                 0 for the first city, since the trip may start anywhere
    goal test  = every chosen city visited
    path cost  = total hours spent travelling between cities

Heuristic h(n) = weight of a minimum spanning tree over {current} + unvisited cities (Prim's algorithm).
Any way of finishing the trip is a path through those cities, and a path is a spanning tree, so it can
never cost less than the MST: h is admissible. It is also consistent (adding the edge current->next to
the MST of the next state gives a spanning tree of this state), so A* with a closed set returns an
optimal order.

For comparison the same search runs with h = 0 (uniform-cost search) and with the weaker "cheapest
edge out of the current city" heuristic; `greedy_order` is the nearest-neighbour baseline, which is
fast but not optimal.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable

INF = float("inf")

Distances = dict[str, dict[str, float]]  # hours between every pair of chosen cities


def mst_weight(nodes: list[str], dist: Distances) -> float:
    """Prim's algorithm. INF if the nodes cannot all be connected."""
    if len(nodes) <= 1:
        return 0.0
    in_tree = {nodes[0]}
    best = {n: dist[nodes[0]].get(n, INF) for n in nodes[1:]}
    total = 0.0
    while best:
        node = min(best, key=best.get)
        if best[node] == INF:
            return INF
        total += best.pop(node)
        in_tree.add(node)
        for other in best:
            best[other] = min(best[other], dist[node].get(other, INF))
    return total


def h_mst(current: str | None, remaining: frozenset[str], dist: Distances) -> float:
    return mst_weight(([current] if current else []) + sorted(remaining), dist)


def h_min_edge(current: str | None, remaining: frozenset[str], dist: Distances) -> float:
    if not remaining or current is None:
        return 0.0
    return min(dist[current].get(r, INF) for r in remaining)


def h_zero(current: str | None, remaining: frozenset[str], dist: Distances) -> float:
    return 0.0


HEURISTICS: dict[str, Callable[[str | None, frozenset[str], Distances], float]] = {
    "mst": h_mst,
    "min_edge": h_min_edge,
    "zero": h_zero,
}
ALGORITHM_NAMES = {"mst": "A* (MST heuristic)", "min_edge": "A* (min-edge heuristic)", "zero": "Uniform-cost search"}


def astar_order(cities: list[str], dist: Distances, heuristic: str = "mst") -> dict:
    """Best visiting order for `cities`.

    Returns:
        dict with order, total_hours, nodes_expanded, nodes_generated, algorithm, found.
    """
    h = HEURISTICS[heuristic]
    goal = frozenset(cities)
    algorithm = ALGORITHM_NAMES[heuristic]
    if len(cities) <= 1:
        return {"order": list(cities), "total_hours": 0.0, "nodes_expanded": 0, "nodes_generated": 0, "algorithm": algorithm, "found": True}

    tie = itertools.count()
    start_h = h(None, goal, dist)
    # (f, g, tie, current, visited, path). Ties break on insertion order, so the result is deterministic.
    frontier = [(start_h, 0.0, next(tie), None, frozenset(), [])]
    closed: set[tuple[str | None, frozenset[str]]] = set()
    expanded = generated = 0

    while frontier:
        f, g, _, current, visited, path = heapq.heappop(frontier)
        if (current, visited) in closed:
            continue
        if visited == goal:
            return {"order": path, "total_hours": round(g, 2), "nodes_expanded": expanded, "nodes_generated": generated, "algorithm": algorithm, "found": True}
        closed.add((current, visited))
        expanded += 1
        for city in sorted(goal - visited):
            step = 0.0 if current is None else dist[current].get(city, INF)
            if step == INF:
                continue
            new_visited = visited | {city}
            if (city, new_visited) in closed:
                continue
            new_g = g + step
            new_h = h(city, goal - new_visited, dist)
            if new_h == INF:
                continue
            generated += 1
            heapq.heappush(frontier, (new_g + new_h, new_g, next(tie), city, new_visited, path + [city]))

    return {"order": list(cities), "total_hours": INF, "nodes_expanded": expanded, "nodes_generated": generated, "algorithm": algorithm, "found": False}


def greedy_order(cities: list[str], dist: Distances) -> dict:
    """Nearest-neighbour baseline: from each possible start, always go to the closest unvisited city."""
    best: dict | None = None
    for start in cities:
        order, total, current = [start], 0.0, start
        left = set(cities) - {start}
        while left:
            nxt = min(sorted(left), key=lambda c: dist[current].get(c, INF))
            total += dist[current].get(nxt, INF)
            order.append(nxt)
            left.remove(nxt)
            current = nxt
        if best is None or total < best["total_hours"]:
            best = {"order": order, "total_hours": round(total, 2)}
    best = best or {"order": [], "total_hours": 0.0}
    return {**best, "nodes_expanded": len(cities) * len(cities), "nodes_generated": 0, "algorithm": "Greedy nearest neighbour", "found": True}
