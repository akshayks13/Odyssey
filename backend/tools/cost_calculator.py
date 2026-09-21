"""Activity, food, and transport cost totals."""
from __future__ import annotations

import math

from langchain_core.tools import tool

from llm import llm_json

_food_cache: dict[tuple[str, str], float] = {}
_DEFAULT_FOOD_PER_DAY = {"budget": 500.0, "mid": 900.0, "premium": 1800.0}  # the one fallback, if the model cannot price it

# Typical Indian road costs in rupees, rough on purpose. A vehicle is priced per vehicle and holds `seats` people;
# a bus seat is priced per person. Tolls are folded into the own-car rate. Local travel is a flat daily allowance.
ROAD_VEHICLES = {
    "own_car": {"label": "own car, fuel and tolls", "per_km": 8.0, "seats": 4},
    "taxi": {"label": "hired taxi", "per_km": 12.0, "seats": 4},
    "tempo_traveller": {"label": "tempo traveller", "per_km": 25.0, "seats": 12},
    "bus": {"label": "bus", "per_km": 2.5, "seats": 1},
}
LOCAL_TRANSPORT_PER_DAY = 500.0  # autos and cabs between sights, per group of four


def _llm_food_cost(region: str, tier: str) -> float | None:
    key = (region.strip().lower(), tier)
    if key not in _food_cache:
        data = llm_json(
            "You estimate travel food costs. Reply ONLY with JSON.",
            f"Typical food spend per person per day (three meals plus snacks) in INR for a {tier}-tier traveller "
            f'in {region}. Schema: {{"per_person_per_day_inr": number}}',
        )
        try:
            value = float(data.get("per_person_per_day_inr") or 0)
        except (TypeError, ValueError):
            value = 0.0
        if not 150 <= value <= 4000:  # outside this range the estimate is not credible
            return None
        _food_cache[key] = value
    return _food_cache[key]


@tool(parse_docstring=True)
def calculate_activity_costs(activity_costs_inr: list[float], travellers: int) -> dict:
    """Sum activity costs across all selected activities for the whole group.

    Args:
        activity_costs_inr: List of per-person activity costs in INR.
        travellers: Number of travellers in the group.

    Returns:
        dict with total_inr (per-person costs already assumed per-head,
        multiplied by travellers).
    """
    total = sum(activity_costs_inr) * travellers
    return {"total_inr": round(total, 2)}


@tool(parse_docstring=True)
def estimate_food_costs(duration_days: int, travellers: int, tier: str = "mid", region: str = "") -> dict:
    """Estimate total food cost for the trip, priced for the region when one is given.

    Args:
        duration_days: Number of days of the trip.
        travellers: Number of travellers.
        tier: One of "budget", "mid", "premium".
        region: Where the trip is, e.g. "Goa, India". Used to price meals locally.

    Returns:
        dict with total_inr and per_person_per_day_inr.
    """
    per_day = (_llm_food_cost(region, tier) if region else None) or _DEFAULT_FOOD_PER_DAY.get(tier, 900.0)
    total = per_day * duration_days * travellers
    return {"total_inr": round(total, 2), "per_person_per_day_inr": per_day}


@tool(parse_docstring=True)
def estimate_road_cost(distance_km: float, travellers: int, vehicle: str = "taxi") -> dict:
    """Price a road journey for the whole group.

    Args:
        distance_km: Road distance one way.
        travellers: Number of travellers.
        vehicle: own_car, taxi, tempo_traveller or bus (anything else is priced as a taxi).

    Returns:
        dict with cost_inr for the group, the vehicle used and how many vehicles (seats for a bus).
    """
    kind = vehicle if vehicle in ROAD_VEHICLES else "taxi"
    spec = ROAD_VEHICLES[kind]
    count = math.ceil(max(1, travellers) / spec["seats"])
    return {"cost_inr": round(distance_km * spec["per_km"] * count, 2), "vehicle": kind, "vehicles": count}


@tool(parse_docstring=True)
def estimate_local_transport(duration_days: int, travellers: int) -> dict:
    """Autos and cabs between the sights over the whole stay (not the journeys between cities).

    Args:
        duration_days: Number of days.
        travellers: Number of travellers.

    Returns:
        dict with total_inr.
    """
    groups = math.ceil(max(1, travellers) / 4)
    return {"total_inr": round(LOCAL_TRANSPORT_PER_DAY * groups * max(1, duration_days), 2)}


@tool(parse_docstring=True)
def calculate_route_cost(leg_costs_inr: list[float], travellers: int) -> dict:
    """Sum per-leg transport costs, scaled for group size where relevant.

    Args:
        leg_costs_inr: List of transport leg costs in INR (assumed per
            vehicle/group already for road trips).
        travellers: Number of travellers (for informational breakdown only;
            road transport cost is not re-multiplied since it is per-vehicle).

    Returns:
        dict with total_inr.
    """
    return {"total_inr": round(sum(leg_costs_inr), 2)}


def reconcile_activity_costs(budget, itinerary, activity_costs: dict[str, float], travellers: int):
    """Return `budget` with the activities line rebuilt from what the schedule actually contains.

    Budget prices a plan before it is scheduled; this replaces that estimate with the real total
    so the number shown to the user (and checked by the Critic) matches the itinerary.
    """
    from models.schemas import BudgetLineItem

    per_person = sum(
        activity_costs.get(item.activity_id, 0.0)
        for day in itinerary.days
        for item in day.items
        if item.kind == "activity"
    )
    activities_total = round(per_person * max(travellers, 1), 2)
    total = round(budget.hotels_inr + budget.food_inr + budget.transport_inr + budget.misc_inr + activities_total, 2)
    items = [li for li in budget.line_items if li.category != "activities"]
    items.insert(min(2, len(items)), BudgetLineItem(category="activities", amount_inr=activities_total))
    return budget.model_copy(
        update={
            "activities_inr": activities_total,
            "total_inr": total,
            "over_budget_by_inr": round(max(0.0, total - budget.ceiling_inr), 2),
            "line_items": items,
        }
    )
