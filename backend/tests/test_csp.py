"""Unit tests for the OR-Tools VRPTW day scheduler."""
from __future__ import annotations

from algorithms.csp_solver import solve_day_schedule


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
