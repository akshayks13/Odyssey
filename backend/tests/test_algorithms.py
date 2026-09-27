"""UCS, A* (MST heuristic) and the CSP scheduler, checked against brute force and hand-worked cases."""
from __future__ import annotations

import itertools

import pytest

from algorithms.astar import astar_order, greedy_order, h_mst, mst_weight
from algorithms.csp_solver import solve_day_schedule
from algorithms.search import build_graph, uniform_cost_search
from tools import world

REGION = "Kerala, India"
CITIES = [c["name"] for c in world.cities(REGION)]


def _dist(cities: list[str]) -> dict[str, dict[str, float]]:
    return {a: {b: world.road_route(REGION, a, b)["hours"] for b in cities if b != a} for a in cities}


def _brute_force(cities: list[str], dist) -> float:
    return min(sum(dist[a][b] for a, b in zip(p, p[1:])) for p in itertools.permutations(cities))


# --- UCS -------------------------------------------------------------------

def test_ucs_finds_the_path_through_junction_towns():
    r = world.road_route(REGION, "Kochi", "Wayanad")
    assert r["found"]
    assert r["path"] == ["Kochi", "Thrissur", "Kozhikode", "Wayanad"]
    assert r["hours"] == pytest.approx(7.5)
    assert r["km"] == pytest.approx(275)


def test_ucs_prefers_a_cheaper_two_hop_path_over_a_dearer_direct_link():
    graph = build_graph([{"a": "A", "b": "C", "h": 10, "k": 1}, {"a": "A", "b": "B", "h": 2, "k": 1}, {"a": "B", "b": "C", "h": 3, "k": 1}], "h", "k")
    r = uniform_cost_search(graph, "A", "C")
    assert r["path"] == ["A", "B", "C"] and r["hours"] == 5


def test_ucs_reports_unreachable_and_trivial_cases():
    graph = build_graph([{"a": "A", "b": "B", "h": 1, "k": 1}, {"a": "C", "b": "D", "h": 1, "k": 1}], "h", "k")
    assert not uniform_cost_search(graph, "A", "D")["found"]
    assert uniform_cost_search(graph, "A", "A")["hours"] == 0


def test_rail_exists_only_between_stations():
    assert world.rail_route(REGION, "Kochi", "Varkala")["found"]
    assert not world.rail_route(REGION, "Kochi", "Munnar")["found"]


# --- A* over visiting order --------------------------------------------------

@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_astar_is_optimal_on_every_subset(size):
    for subset in itertools.combinations(CITIES, size):
        dist = _dist(list(subset))
        result = astar_order(list(subset), dist, "mst")
        assert result["found"]
        assert result["total_hours"] == pytest.approx(_brute_force(list(subset), dist), abs=0.01)


def test_mst_heuristic_never_overestimates():
    cities = CITIES[:6]
    dist = _dist(cities)
    for current in cities:
        rest = [c for c in cities if c != current]
        for k in range(len(rest) + 1):
            for remaining in itertools.combinations(rest, k):
                if not remaining:
                    continue
                true_cost = min(
                    dist[current][p[0]] + sum(dist[a][b] for a, b in zip(p, p[1:]))
                    for p in itertools.permutations(remaining)
                )
                assert h_mst(current, frozenset(remaining), dist) <= true_cost + 1e-9


def test_astar_expands_fewer_nodes_than_ucs_and_beats_greedy_or_ties():
    cities = CITIES[:6]
    dist = _dist(cities)
    a_star, ucs, greedy = astar_order(cities, dist, "mst"), astar_order(cities, dist, "zero"), greedy_order(cities, dist)
    assert a_star["total_hours"] == pytest.approx(ucs["total_hours"])  # both optimal
    assert a_star["nodes_expanded"] < ucs["nodes_expanded"]
    assert greedy["total_hours"] >= a_star["total_hours"] - 1e-9


def test_mst_weight_small_cases():
    dist = {"A": {"B": 1, "C": 4}, "B": {"A": 1, "C": 2}, "C": {"A": 4, "B": 2}}
    assert mst_weight(["A", "B", "C"], dist) == 3
    assert mst_weight(["A"], dist) == 0


def test_astar_single_city_and_determinism():
    assert astar_order(["Kochi"], _dist(["Kochi"]))["order"] == ["Kochi"]
    cities = ["Kochi", "Munnar", "Alleppey", "Varkala"]
    assert astar_order(cities, _dist(cities)) == astar_order(cities, _dist(cities))


# --- CSP scheduler -----------------------------------------------------------

def _act(i, open_h, close_h, minutes, pref=0.5):
    return {"id": f"a{i}", "name": f"Sight {i}", "category": "culture", "duration_minutes": minutes,
            "opening_hour": open_h, "closing_hour": close_h, "preference_score": pref}


def _check_day(result, acts, day_start, day_end, travel=10):
    rows = sorted(result["scheduled"] + result["meals"], key=lambda r: r["start_hour"])
    for a, b in zip(rows, rows[1:]):  # no overlap, room to travel
        assert b["start_hour"] * 60 >= a["end_hour"] * 60 + travel - 1e-6
    by_id = {a["id"]: a for a in acts}
    for r in result["scheduled"]:
        a = by_id[r["id"]]
        assert r["start_hour"] >= max(a["opening_hour"], day_start) - 1e-9
        assert r["end_hour"] <= min(a["closing_hour"], day_end) + 1e-9
    for m in result["meals"]:
        lo, hi = (12, 14) if m["id"] == "_meal_lunch" else (19, 21)
        assert lo <= m["start_hour"] and m["end_hour"] <= hi


def test_csp_respects_opening_hours_meals_and_travel():
    acts = [_act(1, 9, 12, 120), _act(2, 13, 17, 90), _act(3, 17, 20, 60)]
    result = solve_day_schedule(acts, 8, 21)
    assert result["status"] == "FEASIBLE" and len(result["scheduled"]) == 3
    assert {m["id"] for m in result["meals"]} == {"_meal_lunch", "_meal_dinner"}
    _check_day(result, acts, 8, 21)


def test_csp_drops_the_least_wanted_sight_when_the_day_cannot_hold_all():
    acts = [_act(1, 9, 12, 180, pref=0.9), _act(2, 9, 12, 180, pref=0.2)]  # both need the whole morning
    result = solve_day_schedule(acts, 8, 21)
    assert result["selected_ids"] == ["a1"]
    assert result["stats"]["dropped"] == 1


def test_csp_mrv_places_the_tight_sight_first():
    # Earliest-first in list order would put the long flexible sight at 9:00 and block the one that closes at 11;
    # MRV assigns the variable with the smallest domain (the 9-11 sight) first.
    acts = [_act(1, 9, 17, 120, pref=0.9), _act(2, 9, 11, 90, pref=0.8)]
    result = solve_day_schedule(acts, 9, 18, include_meals=False)
    assert len(result["scheduled"]) == 2
    _check_day(result, acts, 9, 18)


def test_csp_late_arrival_schedules_nothing_that_does_not_fit():
    result = solve_day_schedule([_act(1, 6, 9, 90)], 11.5, 18)
    assert result["scheduled"] == []


def test_csp_uses_the_travel_matrix():
    acts = [_act(1, 9, 18, 60), _act(2, 9, 18, 60)]
    matrix = [[0, 30, 30], [30, 0, 90], [30, 90, 0]]
    result = solve_day_schedule(acts, 9, 18, travel_matrix_minutes=matrix, include_meals=False)
    a, b = result["scheduled"]
    assert b["start_hour"] * 60 >= a["end_hour"] * 60 + 90
