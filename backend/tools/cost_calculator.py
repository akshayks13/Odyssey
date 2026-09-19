"""Deterministic cost calculation tools used by the Budget Agent.

No LLM reasoning here by design — arithmetic must be exact (see the
LLM-vs-tool split in the architecture plan).
"""
from __future__ import annotations

from langchain_core.tools import tool

from tools.seed_data import get_food_cost, get_travel_leg


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
def estimate_food_costs(duration_days: int, travellers: int, tier: str = "mid") -> dict:
    """Estimate total food cost for the trip.

    Args:
        duration_days: Number of days of the trip.
        travellers: Number of travellers.
        tier: One of "budget", "mid", "premium".

    Returns:
        dict with total_inr and per_person_per_day_inr.
    """
    per_day = get_food_cost(tier)
    total = per_day * duration_days * travellers
    return {"total_inr": round(total, 2), "per_person_per_day_inr": per_day}


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


@tool(parse_docstring=True)
def check_transport_availability(origin: str, destination: str, mode: str = "road") -> dict:
    """Rules engine for bus/train/taxi/air availability between two places.

    Args:
        origin: Origin city name.
        destination: Destination city name.
        mode: One of road, rail, air.

    Returns:
        dict with available (bool), mode, reason.
    """
    mode = (mode or "road").lower()
    if origin.strip().lower() == destination.strip().lower():
        return {"available": False, "mode": mode, "reason": "same city"}
    if mode == "air":
        return {
            "available": False,
            "mode": "air",
            "reason": "Intra-region hops use road; call search_flights only for distinct IATA cities",
        }
    if mode == "rail":
        hill = {"munnar", "thekkady", "vagamon", "wayanad"}
        if origin.strip().lower() in hill or destination.strip().lower() in hill:
            return {"available": False, "mode": "rail", "reason": "no direct rail to hill station — use road"}
        return {"available": True, "mode": "rail", "reason": "regional rail corridor"}
    leg = get_travel_leg(origin, destination)
    if leg:
        return {
            "available": True,
            "mode": leg.get("mode", "road"),
            "reason": "seed corridor",
            "duration_hours": leg["duration_hours"],
        }
    return {"available": True, "mode": "road", "reason": "road taxi/bus assumed"}
