"""Rupee arithmetic: road fares, food, local transport, activities."""
from __future__ import annotations

import math

from tools import world

# Rough Indian road costs. A vehicle holds `seats` people and is priced per vehicle; a bus seat per person.
ROAD_VEHICLES = {
    "taxi": {"label": "hired taxi", "per_km": 12.0, "seats": 4},
    "tempo_traveller": {"label": "tempo traveller", "per_km": 25.0, "seats": 12},
    "bus": {"label": "bus", "per_km": 2.5, "seats": 1},
}
LOCAL_TRANSPORT_PER_DAY = 500.0  # autos and cabs between sights, per group of four


def road_vehicle_for(travellers: int) -> str:
    """A taxi for up to four people, a tempo traveller for a bigger group."""
    return "tempo_traveller" if travellers > 4 else "taxi"


def estimate_road_cost(distance_km: float, travellers: int, vehicle: str = "taxi") -> float:
    spec = ROAD_VEHICLES.get(vehicle, ROAD_VEHICLES["taxi"])
    count = math.ceil(max(1, travellers) / spec["seats"])
    return round(distance_km * spec["per_km"] * count, 2)


def estimate_food_costs(region: str, duration_days: int, travellers: int, tier: str) -> dict:
    per_day = world.food_per_day(region, tier)
    return {"total_inr": round(per_day * duration_days * travellers, 2), "per_person_per_day_inr": per_day}


def estimate_local_transport(duration_days: int, travellers: int) -> float:
    groups = math.ceil(max(1, travellers) / 4)
    return round(LOCAL_TRANSPORT_PER_DAY * groups * max(1, duration_days), 2)


def calculate_activity_costs(per_person_costs: list[float], travellers: int) -> float:
    return round(sum(per_person_costs) * travellers, 2)


def reconcile_activity_costs(budget, itinerary, activity_costs: dict[str, float], travellers: int):
    """`budget` with the activities line rebuilt from what the schedule actually contains.

    Budget prices a plan before it is scheduled; this replaces that estimate with the real total so the
    number shown (and checked by the Critic) matches the itinerary.
    """
    from models.schemas import BudgetLineItem

    per_person = sum(activity_costs.get(i.activity_id, 0.0) for d in itinerary.days for i in d.items if i.kind == "activity")
    activities_total = round(per_person * max(travellers, 1), 2)
    total = round(budget.hotels_inr + budget.food_inr + budget.transport_inr + activities_total, 2)
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
