"""Deterministic budget validation tool used by the Budget Agent and Critic."""
from __future__ import annotations

from langchain_core.tools import tool


@tool(parse_docstring=True)
def validate_budget(total_inr: float, ceiling_inr: float) -> dict:
    """Compare an itemized total against the budget ceiling.

    Args:
        total_inr: Total itemized cost in INR.
        ceiling_inr: The user's budget ceiling in INR.

    Returns:
        dict with within_budget (bool) and over_by_inr (float, 0 if within).
    """
    over = max(0.0, total_inr - ceiling_inr)
    return {"within_budget": over == 0.0, "over_by_inr": round(over, 2)}


@tool(parse_docstring=True)
def generate_tradeoff_options(over_by_inr: float, hotel_price_diff_inr: float, activity_cost_inr: float) -> dict:
    """Suggest concrete ways to close a budget gap, ranked by damage to the plan.

    Args:
        over_by_inr: Amount the plan currently exceeds budget by (INR).
        hotel_price_diff_inr: Savings available by switching to a cheaper hotel (INR/night, will be
            multiplied by nights by the caller before comparing).
        activity_cost_inr: Cost of the single lowest-preference-score activity that could be dropped.

    Returns:
        dict with ranked list of suggestion strings.
    """
    options = []
    if hotel_price_diff_inr > 0:
        options.append(f"Switch to a cheaper hotel to save ~₹{hotel_price_diff_inr:.0f}")
    if activity_cost_inr > 0:
        options.append(f"Drop the lowest-preference-score activity to save ~₹{activity_cost_inr:.0f}")
    options.append("Reduce trip by one day to cut hotel + food costs")
    return {"over_by_inr": over_by_inr, "suggestions": options}
