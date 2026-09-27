"""Budget checks used by the Budget agent and the Critic."""
from __future__ import annotations


def validate_budget(total_inr: float, ceiling_inr: float) -> dict:
    """within_budget, and how far over the ceiling the total is (0 when within)."""
    over = max(0.0, total_inr - ceiling_inr)
    return {"within_budget": over == 0.0, "over_by_inr": round(over, 2)}


def generate_tradeoff_options(over_by_inr: float, hotel_price_diff_inr: float, activity_cost_inr: float) -> list[str]:
    """Ways to close a budget gap, least damaging first."""
    options = []
    if hotel_price_diff_inr > 0:
        options.append(f"Switch to cheaper hotels to save about ₹{hotel_price_diff_inr:,.0f}")
    if activity_cost_inr > 0:
        options.append(f"Drop the priciest remaining activity to save about ₹{activity_cost_inr:,.0f}")
    options.append("Shorten the trip by one day to cut hotel and food costs")
    return options
