"""Uniform-cost search (Dijkstra) over a weighted, undirected network.

Used by Mobility to turn the sparse road and rail networks into city-to-city travel times: Kochi to
Wayanad has no direct road, so the cheapest path goes Kochi -> Thrissur -> Kozhikode -> Wayanad.

Problem formulation:
    state      = the town we are in
    actions    = drive (or ride) along one road / rail link to a neighbouring town
    step cost  = hours for that link (always > 0)
    goal test  = state == destination
UCS expands the frontier node with the lowest path cost g(n). With positive step costs it is complete and
optimal, and the first time the goal is popped its g is the shortest travel time.
"""
from __future__ import annotations

import heapq
import itertools

# graph: {town: [(neighbour, hours, km_or_fare), ...]}
Graph = dict[str, list[tuple[str, float, float]]]


def build_graph(links: list[dict], weight: str, extra: str) -> Graph:
    """Undirected adjacency list from link rows ({"a", "b", weight, extra})."""
    graph: Graph = {}
    for link in links:
        a, b = link["a"], link["b"]
        graph.setdefault(a, []).append((b, float(link[weight]), float(link[extra])))
        graph.setdefault(b, []).append((a, float(link[weight]), float(link[extra])))
    return graph


def uniform_cost_search(graph: Graph, start: str, goal: str) -> dict:
    """Cheapest path from `start` to `goal`.

    Returns:
        dict with found, hours (path cost), extra (the summed second weight: km for roads, fare for
        rail), path (list of towns) and nodes_expanded.
    """
    if start == goal:
        return {"found": True, "hours": 0.0, "extra": 0.0, "path": [start], "nodes_expanded": 0}
    if start not in graph or goal not in graph:
        return {"found": False, "hours": float("inf"), "extra": 0.0, "path": [], "nodes_expanded": 0}

    tie = itertools.count()
    frontier: list[tuple[float, int, str, float, list[str]]] = [(0.0, next(tie), start, 0.0, [start])]
    explored: set[str] = set()
    nodes_expanded = 0

    while frontier:
        g, _, town, extra, path = heapq.heappop(frontier)
        if town in explored:
            continue  # a cheaper copy of this state was already expanded
        if town == goal:  # goal test on expansion, not generation: that is what makes UCS optimal
            return {"found": True, "hours": round(g, 2), "extra": round(extra, 2), "path": path, "nodes_expanded": nodes_expanded}
        explored.add(town)
        nodes_expanded += 1
        for neighbour, hours, more in graph.get(town, []):
            if neighbour not in explored:
                heapq.heappush(frontier, (g + hours, next(tie), neighbour, extra + more, path + [neighbour]))

    return {"found": False, "hours": float("inf"), "extra": 0.0, "path": [], "nodes_expanded": nodes_expanded}
