"""Weighted A* over visit order: state = (city, visited set, elapsed hours today)."""
from __future__ import annotations

import heapq
import itertools

import networkx as nx


def build_travel_graph(destinations: list[str], get_leg) -> nx.DiGraph:
    """Build a directed graph where edges hold (time, cost, distance, mode).

    Args:
        destinations: list of destination names (including the trip's start city).
        get_leg: callable(origin, destination) -> dict with duration_hours,
            cost_inr, distance_km, mode.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(destinations)
    for origin, dest in itertools.permutations(destinations, 2):
        leg = get_leg(origin, dest)
        graph.add_edge(
            origin,
            dest,
            time=leg["duration_hours"],
            cost=leg["cost_inr"],
            distance=leg["distance_km"],
            mode=leg.get("mode", "road"),
        )
    return graph


def _heuristic(graph: nx.DiGraph, current: str, remaining: frozenset[str]) -> float:
    if not remaining:
        return 0.0
    reachable = [graph[current][r]["time"] for r in remaining if graph.has_edge(current, r)]
    return min(reachable) if reachable else 0.0


def astar_route_search(
    graph: nx.DiGraph,
    start: str,
    must_visit: list[str],
    max_expansions: int = 4000,
    max_daily_travel_hours: float = 4.0,
) -> dict:
    """Find the travel-time-optimal order to visit all `must_visit` cities
    starting from `start`.

    State is `(current_city, frozenset(visited), elapsed_hours_today)` so a
    hop that would exceed `max_daily_travel_hours` is either taken as an
    overnight (elapsed resets) or, if the hop itself is over the cap,
    heavily penalized.

    Returns:
        dict with order, total_time_hours, total_cost_inr, total_distance_km,
        nodes_expanded, algorithm.
    """
    goal_set = frozenset(must_visit) - {start}
    if not goal_set:
        return {
            "order": [start],
            "total_time_hours": 0.0,
            "total_cost_inr": 0.0,
            "total_distance_km": 0.0,
            "nodes_expanded": 0,
            "algorithm": "weighted_astar",
        }

    cap = max(0.5, float(max_daily_travel_hours))
    over_cap_penalty = 50.0
    counter = itertools.count()
    epsilon = 0.0
    nodes_expanded = 0
    best_g: dict[tuple[str, frozenset, int], float] = {}

    start_visited = frozenset()
    start_h = _heuristic(graph, start, goal_set)
    # heap: (f, g, tie, current, visited, elapsed_today, path, cost, distance)
    open_heap: list[tuple] = [
        (start_h, 0.0, next(counter), start, start_visited, 0.0, [start], 0.0, 0.0)
    ]

    best_partial: dict | None = None

    while open_heap:
        f, g, _, current, visited, elapsed, path, cost, dist = heapq.heappop(open_heap)
        nodes_expanded += 1

        state_key = (current, visited, int(elapsed * 10))
        if state_key in best_g and best_g[state_key] <= g:
            continue
        best_g[state_key] = g

        if visited == goal_set:
            return {
                "order": path,
                "total_time_hours": round(g, 2),
                "total_cost_inr": round(cost, 2),
                "total_distance_km": round(dist, 1),
                "nodes_expanded": nodes_expanded,
                "algorithm": "weighted_astar",
            }

        if best_partial is None or len(visited) > len(best_partial["visited"]):
            best_partial = {
                "path": path,
                "visited": visited,
                "g": g,
                "cost": cost,
                "dist": dist,
                "elapsed": elapsed,
            }

        if nodes_expanded >= max_expansions:
            epsilon = min(epsilon + 4.0, 20.0)

        if nodes_expanded > max_expansions * 3:
            break

        for neighbor in goal_set - visited:
            if not graph.has_edge(current, neighbor):
                continue
            edge = graph[current][neighbor]
            hop = float(edge["time"])
            penalty = over_cap_penalty if hop > cap else 0.0
            if elapsed + hop <= cap:
                new_elapsed = elapsed + hop
            else:
                # new calendar day, then travel
                new_elapsed = hop if hop <= cap else cap
            new_g = g + hop
            new_visited = visited | {neighbor}
            new_h = _heuristic(graph, neighbor, goal_set - new_visited)
            new_f = new_g + penalty + (1 + epsilon) * new_h
            heapq.heappush(
                open_heap,
                (
                    new_f,
                    new_g,
                    next(counter),
                    neighbor,
                    new_visited,
                    new_elapsed,
                    path + [neighbor],
                    cost + edge["cost"],
                    dist + edge["distance"],
                ),
            )

    if best_partial:
        path = list(best_partial["path"])
        visited = set(best_partial["visited"])
        current = path[-1]
        g, cost, dist = best_partial["g"], best_partial["cost"], best_partial["dist"]
        for remaining in goal_set - visited:
            if graph.has_edge(current, remaining):
                edge = graph[current][remaining]
                g += edge["time"]
                cost += edge["cost"]
                dist += edge["distance"]
                path.append(remaining)
                current = remaining
        return {
            "order": path,
            "total_time_hours": round(g, 2),
            "total_cost_inr": round(cost, 2),
            "total_distance_km": round(dist, 1),
            "nodes_expanded": nodes_expanded,
            "algorithm": "weighted_astar_anytime_fallback",
        }

    return {
        "order": [start, *must_visit],
        "total_time_hours": 0.0,
        "total_cost_inr": 0.0,
        "total_distance_km": 0.0,
        "nodes_expanded": nodes_expanded,
        "algorithm": "trivial_fallback",
    }
