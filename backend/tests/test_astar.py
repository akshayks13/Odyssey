"""Unit tests for weighted A* destination-ordering search."""
from __future__ import annotations

from algorithms.astar import astar_route_search, build_travel_graph


def _leg_table():
    # Symmetric travel times (hours) for a small Kerala subgraph.
    table = {
        ("Kochi", "Munnar"): {"duration_hours": 4.0, "cost_inr": 900, "distance_km": 130, "mode": "road"},
        ("Kochi", "Thekkady"): {"duration_hours": 5.0, "cost_inr": 1200, "distance_km": 190, "mode": "road"},
        ("Kochi", "Alleppey"): {"duration_hours": 1.5, "cost_inr": 500, "distance_km": 60, "mode": "road"},
        ("Munnar", "Thekkady"): {"duration_hours": 3.0, "cost_inr": 700, "distance_km": 100, "mode": "road"},
        ("Munnar", "Alleppey"): {"duration_hours": 4.5, "cost_inr": 1000, "distance_km": 160, "mode": "road"},
        ("Thekkady", "Alleppey"): {"duration_hours": 3.5, "cost_inr": 900, "distance_km": 140, "mode": "road"},
    }
    reverse = {(b, a): v for (a, b), v in table.items()}
    table.update(reverse)
    return table


def _get_leg(origin, dest):
    return _leg_table()[(origin, dest)]


def test_build_travel_graph_is_complete():
    cities = ["Kochi", "Munnar", "Thekkady"]
    graph = build_travel_graph(cities, _get_leg)
    assert set(graph.nodes) == set(cities)
    # Directed complete graph minus self-loops: n*(n-1)
    assert graph.number_of_edges() == 6
    assert graph["Kochi"]["Munnar"]["time"] == 4.0


def test_astar_single_city_is_trivial():
    graph = build_travel_graph(["Kochi"], _get_leg)
    result = astar_route_search(graph, "Kochi", ["Kochi"])
    assert result["order"] == ["Kochi"]
    assert result["total_time_hours"] == 0.0
    assert result["nodes_expanded"] == 0


def test_astar_finds_shortest_order():
    cities = ["Kochi", "Munnar", "Thekkady", "Alleppey"]
    graph = build_travel_graph(cities, _get_leg)
    result = astar_route_search(graph, "Kochi", cities)

    assert result["order"][0] == "Kochi"
    assert set(result["order"]) == set(cities)
    assert result["algorithm"] == "weighted_astar"
    assert result["nodes_expanded"] > 0
    # The cheapest tour starting at Kochi visiting all three others:
    # Kochi -> Alleppey (1.5) -> Thekkady (3.5) -> Munnar (3.0) = 8.0
    # or Kochi -> Munnar (4.0) -> Thekkady (3.0) -> Alleppey (3.5) = 10.5
    # so the optimal is Alleppey first.
    assert result["order"][1] == "Alleppey"
    assert result["total_time_hours"] == 8.0


def test_astar_always_returns_a_plan_even_with_tiny_expansion_budget():
    cities = ["Kochi", "Munnar", "Thekkady", "Alleppey"]
    graph = build_travel_graph(cities, _get_leg)
    result = astar_route_search(graph, "Kochi", cities, max_expansions=1)
    assert set(result["order"]) == set(cities)
    assert result["algorithm"] in {"weighted_astar", "weighted_astar_anytime_fallback"}


def test_astar_respects_daily_travel_cap_parameter():
    cities = ["Kochi", "Munnar", "Thekkady", "Alleppey"]
    graph = build_travel_graph(cities, _get_leg)
    result = astar_route_search(graph, "Kochi", cities, max_daily_travel_hours=4.0)
    assert result["order"][0] == "Kochi"
    assert set(result["order"]) == set(cities)
    assert result["total_time_hours"] >= 8.0
