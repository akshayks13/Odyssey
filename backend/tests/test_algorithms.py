"""Pure algorithms: A* city order, the day scheduler, stay plan, preference and cost scorers."""
from __future__ import annotations

import pytest

from algorithms.astar import astar_route_search, build_travel_graph
from algorithms.csp_solver import solve_day_schedule
from algorithms.planning import plan_stays
from algorithms.optimizer import compute_score, infer_archetype
from tools.budget_validator import validate_budget
from tools.preference_scorer import cosine_similarity, score_preference_match
from tools.schedule_validator import check_schedule_conflicts, validate_time_windows


# ====================================================================================================
# A* city ordering
# ====================================================================================================

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


# ====================================================================================================
# Day scheduler (OR-Tools VRPTW)
# ====================================================================================================

MUNNAR_ACTIVITIES = [
    {
        "id": "munnar_tea",
        "name": "Tea Plantation Tour",
        "category": "nature",
        "duration_minutes": 150,
        "opening_hour": 9,
        "closing_hour": 17,
        "preference_score": 0.9,
    },
    {
        "id": "munnar_eravikulam",
        "name": "Eravikulam National Park",
        "category": "nature",
        "duration_minutes": 180,
        "opening_hour": 8,
        "closing_hour": 16,
        "preference_score": 0.9,
    },
    {
        "id": "munnar_trek",
        "name": "Trekking Trail",
        "category": "adventure",
        "duration_minutes": 240,
        "opening_hour": 7,
        "closing_hour": 15,
        "preference_score": 0.85,
    },
    {
        "id": "munnar_view",
        "name": "Top Station Viewpoint",
        "category": "nature",
        "duration_minutes": 90,
        "opening_hour": 8,
        "closing_hour": 18,
        "preference_score": 0.75,
    },
]


def test_empty_pool_returns_no_activities():
    result = solve_day_schedule([])
    assert result["scheduled"] == []
    assert result["status"] == "NO_ACTIVITIES"


def test_schedule_is_feasible_and_non_overlapping():
    result = solve_day_schedule(MUNNAR_ACTIVITIES)
    assert result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(result["scheduled"]) >= 1

    items = sorted(result["scheduled"], key=lambda s: s["start_hour"])
    for prev, nxt in zip(items, items[1:]):
        assert prev["end_hour"] <= nxt["start_hour"] + 1e-6

    for item in items:
        assert item["end_hour"] > item["start_hour"]


def test_schedule_respects_opening_hours():
    result = solve_day_schedule(MUNNAR_ACTIVITIES)
    by_id = {a["id"]: a for a in MUNNAR_ACTIVITIES}
    for item in result["scheduled"]:
        source = by_id[item["id"]]
        assert item["start_hour"] >= source["opening_hour"] - 1e-6
        assert item["end_hour"] <= source["closing_hour"] + 1e-6


def test_impossible_activity_is_excluded():
    impossible = [
        {
            "id": "too_long",
            "name": "Impossible 20h hike",
            "category": "adventure",
            "duration_minutes": 1200,
            "opening_hour": 9,
            "closing_hour": 11,
            "preference_score": 1.0,
        }
    ]
    result = solve_day_schedule(impossible)
    assert "too_long" not in result["selected_ids"]


def test_vrptw_with_meals_still_schedules_real_activities():
    result = solve_day_schedule(MUNNAR_ACTIVITIES, include_meals=True)
    assert result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(result["scheduled"]) >= 1
    assert all(not s["id"].startswith("_meal") for s in result["scheduled"])


def test_vrptw_uses_travel_matrix():
    acts = MUNNAR_ACTIVITIES[:2]
    # depot + 2 activities
    matrix = [
        [0, 20, 40],
        [20, 0, 15],
        [40, 15, 0],
    ]
    result = solve_day_schedule(acts, travel_matrix_minutes=matrix, include_meals=False)
    assert result["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(result["scheduled"]) >= 1


# ====================================================================================================
# Scheduler and stay plan on realistic days
# ====================================================================================================

def _acts(n, dur=120, opening=9, closing=18, pref=0.7):
    return [
        {"id": f"a{i}", "name": f"Sight {i}", "category": "nature", "duration_minutes": dur,
         "opening_hour": opening, "closing_hour": closing, "preference_score": pref, "rating": 4.2}
        for i in range(n)
    ]


@pytest.mark.parametrize(
    "n,dur,opening,closing,start,end",
    [(4, 180, 10, 17, 7.0, 22.0), (3, 120, 10, 18, 7.0, 22.0), (4, 120, 11, 19, 7.0, 22.0), (3, 150, 11, 18, 8.0, 21.0)],
)
def test_sights_that_open_late_do_not_empty_the_day(n, dur, opening, closing, start, end):
    """A 90-minute waiting cap made every sight unreachable when they opened well after the day began."""
    result = solve_day_schedule(_acts(n, dur, opening, closing), day_start_hour=start, day_end_hour=end)
    assert result["scheduled"], result["status"]
    for item in result["scheduled"]:
        assert item["start_hour"] >= opening - 1e-6 and item["end_hour"] <= closing + 1e-6


def test_the_morning_is_used():
    result = solve_day_schedule(_acts(2, 120, 9, 18), day_start_hour=9.0, day_end_hour=18.0)
    assert min(s["start_hour"] for s in result["scheduled"]) <= 10.5


def test_meals_are_separate_from_sights_and_never_overlap_them():
    result = solve_day_schedule(_acts(2, 90, 9, 18), day_start_hour=8.0, day_end_hour=21.0)
    assert result["meals"] and all(not s["id"].startswith("_meal") for s in result["scheduled"])
    everything = sorted([*result["scheduled"], *result["meals"]], key=lambda x: x["start_hour"])
    for a, b in zip(everything, everything[1:]):
        assert a["end_hour"] <= b["start_hour"] + 1e-6


def test_a_far_off_city_centre_does_not_block_the_day():
    """The depot stands in for the hotel; a centre 250 km from the sights used to make every day infeasible."""
    n = 3
    matrix = [[0 if i == j else (240 if 0 in (i, j) else 8) for j in range(n + 1)] for i in range(n + 1)]
    assert solve_day_schedule(_acts(n, 90, 9, 18), travel_matrix_minutes=matrix)["scheduled"]


def test_travel_matrix_is_remapped_when_an_activity_cannot_fit():
    acts = _acts(3, 120, 9, 18)
    acts[1]["closing_hour"] = 9  # can never fit
    matrix = [[0, 20, 20, 20], [20, 0, 5, 5], [20, 5, 0, 5], [20, 5, 5, 0]]
    result = solve_day_schedule(acts, travel_matrix_minutes=matrix, include_meals=False, day_end_hour=22.0)
    assert "a1" not in result["selected_ids"] and len(result["scheduled"]) == 2


def test_stay_plan_covers_the_trip():
    blocks = plan_stays(["A", "B", "C"], 7, {"A": 0, "B": 2, "C": 3}, {"A": 6, "B": 6, "C": 6})
    assert sum(b.days for b in blocks) == 7 and sum(b.nights for b in blocks) == 6  # no hotel on the last day


def test_a_long_transfer_costs_a_travel_day():
    blocks = plan_stays(["Vagamon", "Wayanad"], 5, {"Wayanad": 10.9}, {"Vagamon": 7, "Wayanad": 8})
    assert next(b for b in blocks if b.destination == "Wayanad").travel_day
    assert sum(b.days for b in blocks) == 5


def test_a_city_with_little_left_to_see_does_not_get_extra_days():
    assert next(b for b in plan_stays(["A", "B"], 6, {}, {"A": 2, "B": 12}) if b.destination == "A").days <= 2


def test_more_cities_than_days_never_exceeds_the_trip():
    assert sum(b.days for b in plan_stays(["A", "B", "C", "D"], 2, {}, {})) == 2


# ====================================================================================================
# Scorers and validators
# ====================================================================================================

def test_cosine_identical_vectors_is_one():
    vec = {"nature": 1.0, "adventure": 1.0, "food": 0.0, "nightlife": 0.0, "relaxation": 0.0, "culture": 0.0, "shopping": 0.0}
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors_is_zero():
    a = {"nature": 1.0, "adventure": 0.0, "food": 0.0, "nightlife": 0.0, "relaxation": 0.0, "culture": 0.0, "shopping": 0.0}
    b = {"nature": 0.0, "adventure": 0.0, "food": 0.0, "nightlife": 0.0, "relaxation": 0.0, "culture": 0.0, "shopping": 1.0}
    assert cosine_similarity(a, b) == 0.0


def test_score_preference_match_tool_clips_to_unit_interval():
    result = score_preference_match.invoke(
        {
            "preferences": {"nature": 0.9, "adventure": 0.8},
            "category_scores": {"nature": 0.95, "adventure": 0.75, "food": 0.2},
        }
    )
    assert 0.0 <= result["score"] <= 1.0
    assert result["score"] > 0.7


def test_validate_budget_within():
    result = validate_budget.invoke({"total_inr": 34000, "ceiling_inr": 40000})
    assert result["within_budget"] is True
    assert result["over_by_inr"] == 0.0


def test_validate_budget_over():
    result = validate_budget.invoke({"total_inr": 45000, "ceiling_inr": 40000})
    assert result["within_budget"] is False
    assert result["over_by_inr"] == 5000.0


def test_schedule_overlap_detected():
    result = validate_time_windows.invoke(
        {
            "scheduled_items": [
                {"start_hour": 9.0, "end_hour": 12.0, "activity_name": "A"},
                {"start_hour": 11.0, "end_hour": 13.0, "activity_name": "B"},
            ]
        }
    )
    assert result["valid"] is False
    assert len(result["conflicts"]) == 1


def test_schedule_no_overlap_is_valid():
    result = validate_time_windows.invoke(
        {
            "scheduled_items": [
                {"start_hour": 9.0, "end_hour": 11.0, "activity_name": "A"},
                {"start_hour": 11.0, "end_hour": 13.0, "activity_name": "B"},
            ]
        }
    )
    assert result["valid"] is True


def test_overpacked_day_flagged():
    result = check_schedule_conflicts.invoke(
        {
            "days": [
                {
                    "day_number": 1,
                    "items": [
                        {"start_hour": 6.0, "end_hour": 12.0, "activity_name": "A"},
                        {"start_hour": 12.0, "end_hour": 21.0, "activity_name": "B"},
                    ],
                }
            ]
        }
    )
    assert result["valid"] is False
    assert any("overpacked" in i["message"] for i in result["issues"])


def test_infer_archetype_budget_vs_adventure():
    assert infer_archetype({"adventure": 0.9, "relaxation": 0.2}, budget_inr=10000, travellers=2) == "budget"
    assert infer_archetype({"adventure": 0.9, "relaxation": 0.2}, budget_inr=50000, travellers=2) == "adventure"
    assert infer_archetype({"adventure": 0.2, "relaxation": 0.9}, budget_inr=50000, travellers=2) == "relaxed"


def test_compute_score_clips_and_uses_weights():
    result = compute_score(
        preference_satisfaction=1.0,
        activity_quality=1.0,
        route_efficiency=1.0,
        budget_efficiency=1.0,
        travel_burden=0.0,
        constraint_violations=0.0,
        archetype="balanced",
    )
    assert 0.0 <= result["total"] <= 1.0
    assert result["archetype"] == "balanced"
    assert "preference_satisfaction" in result["breakdown"]
