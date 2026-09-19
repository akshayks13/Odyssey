"""
Multi-objective itinerary scoring.

    Score = w_p*P + w_q*Q + w_r*R + w_b*B - w_t*T - w_c*C

  P = preference satisfaction (avg cosine match of scheduled activities)
  Q = activity quality (avg rating, normalized)
  R = route efficiency (inverse of travel-time-to-activity-time ratio)
  B = budget efficiency (1 - overspend fraction, clipped at 0)
  T = travel burden (fraction of total trip time spent travelling)
  C = constraint violations (count of unresolved Critic issues, normalized)

Weights are personalized per traveller archetype, inferred from the
PreferenceWeights the Trip Analyst extracted (adventure vs relaxed vs
budget-conscious travellers value different terms).
"""
from __future__ import annotations

ARCHETYPE_WEIGHTS = {
    "adventure": {"p": 0.30, "q": 0.20, "r": 0.10, "b": 0.10, "t": 0.05, "c": 0.25},
    "relaxed": {"p": 0.25, "q": 0.15, "r": 0.15, "b": 0.10, "t": 0.25, "c": 0.10},
    "budget": {"p": 0.20, "q": 0.10, "r": 0.15, "b": 0.35, "t": 0.10, "c": 0.10},
    "balanced": {"p": 0.25, "q": 0.20, "r": 0.15, "b": 0.15, "t": 0.15, "c": 0.10},
}


def infer_archetype(preferences: dict[str, float], budget_inr: float, travellers: int) -> str:
    """Cheap heuristic archetype inference used to pick score weights."""
    adventure_score = preferences.get("adventure", 0.5)
    relaxation_score = preferences.get("relaxation", 0.5)
    per_person_budget = budget_inr / max(travellers, 1)

    if per_person_budget < 8000:
        return "budget"
    if adventure_score >= 0.7 and adventure_score > relaxation_score:
        return "adventure"
    if relaxation_score >= 0.7:
        return "relaxed"
    return "balanced"


def compute_score(
    preference_satisfaction: float,
    activity_quality: float,
    route_efficiency: float,
    budget_efficiency: float,
    travel_burden: float,
    constraint_violations: float,
    archetype: str = "balanced",
) -> dict:
    """Compute the weighted multi-objective score.

    All input components are expected in [0, 1] except constraint_violations
    which is a raw penalty value in [0, 1] (already normalized by caller).

    Returns:
        dict with total (float, clipped to [0,1]) and breakdown per term.
    """
    w = ARCHETYPE_WEIGHTS.get(archetype, ARCHETYPE_WEIGHTS["balanced"])

    positive = (
        w["p"] * preference_satisfaction
        + w["q"] * activity_quality
        + w["r"] * route_efficiency
        + w["b"] * budget_efficiency
    )
    negative = w["t"] * travel_burden + w["c"] * constraint_violations
    total = max(0.0, min(1.0, positive - negative))

    return {
        "total": round(total, 4),
        "archetype": archetype,
        "breakdown": {
            "preference_satisfaction": round(preference_satisfaction, 4),
            "activity_quality": round(activity_quality, 4),
            "route_efficiency": round(route_efficiency, 4),
            "budget_efficiency": round(budget_efficiency, 4),
            "travel_burden_penalty": round(travel_burden, 4),
            "constraint_violation_penalty": round(constraint_violations, 4),
        },
        "weights": w,
    }
