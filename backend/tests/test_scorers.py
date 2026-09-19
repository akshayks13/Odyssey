"""Unit tests for preference scoring, budget validation, and multi-objective optimizer."""
from __future__ import annotations

from algorithms.optimizer import compute_score, infer_archetype
from tools.budget_validator import validate_budget
from tools.preference_scorer import cosine_similarity, score_preference_match
from tools.schedule_validator import check_schedule_conflicts, validate_time_windows


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
