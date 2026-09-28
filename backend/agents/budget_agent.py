"""Budget Optimization — price hotels, food, sights and transport against the ceiling.

Role: money optimizer. Runs after Mobility so the fares are known. Does not reorder cities or schedule.

Rules:
  - Hotel tier (and food) from the budget per person: under ₹8,000 -> budget, over ₹20,000 -> premium,
    else mid. If the last plan priced did not fit the ceiling, one tier cheaper than that plan used.
  - A hotel the traveller asked for ("cheapest", "best", or a name) always wins over the tier.
  - Days and nights per city come from the shared stay plan, so the bill matches the schedule.
  - Still over the ceiling, cheapest damage first: switch hotels the traveller did not choose to the cheapest
    one (biggest saving first); then greedy trimming as in the fractional-knapsack heuristic — drop the paid
    sight with the lowest preference per rupee first, keeping at least one sight per city.
"""
from __future__ import annotations

import math

from agents.base import Agent, Post, Result
from algorithms.planning import PACE_CAPS, activity_is_excluded, plan_stays
from core.messages import Message, MsgType
from models.schemas import BudgetBreakdown, BudgetLineItem, DisruptionType, EditLocks, Hotel
from tools import world
from tools.budget_validator import generate_tradeoff_options, validate_budget
from tools.cost_calculator import calculate_activity_costs, estimate_food_costs, estimate_local_transport

ROOM_SIZE = 2  # travellers per room
TIERS = ["budget", "mid", "premium"]


def budget_tier(per_person_inr: float) -> str:
    if per_person_inr < 8000:
        return "budget"
    if per_person_inr > 20000:
        return "premium"
    return "mid"


def pick_hotel(rows: list[dict], city: str, tier: str, locks: EditLocks) -> tuple[dict, bool]:
    """One hotel for a city, and whether the traveller chose it (then budget cuts leave it alone)."""
    pref = locks.hotel_prefs.get(city.lower()) or locks.hotel_prefs.get("*")
    if pref == "cheapest":
        return min(rows, key=lambda h: h["price_per_night_inr"]), True
    if pref == "best":
        return max(rows, key=lambda h: (h["rating"], -h["price_per_night_inr"])), True
    named = next((h for h in rows if pref and pref.lower() in h["name"].lower()), None)
    if named:
        return named, True
    return next((h for h in rows if h["tier"] == tier), sorted(rows, key=lambda h: h["price_per_night_inr"])[0]), False


def price_trip(state: dict) -> Result:
    """The whole pricing procedure, from the blackboard view to the agent's result."""
    spec = state["trip_spec"]
    region = spec.destination_region
    selected = state.get("selected_destinations", [])
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    previous = state.get("budget_breakdown")

    ceiling = spec.budget_inr
    for d in state.get("disruptions", []):
        if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr:
            ceiling = d.new_budget_inr

    tier = budget_tier(ceiling / max(spec.travellers, 1))
    cheaper = previous is not None and previous.total_inr > ceiling  # the last priced plan did not fit this ceiling
    if cheaper:
        prev = min((TIERS.index(h.tier) for h in previous.selected_hotels if h.tier in TIERS), default=TIERS.index(tier))
        tier = TIERS[max(0, min(prev, TIERS.index(tier)) - 1)]

    order = route.ordered_destinations if route and route.ordered_destinations else [d.name for d in selected]
    inbound = {leg.destination: leg.duration_hours for leg in (route.legs if route else [])}
    stay_plan = plan_stays(order, spec.duration_days, inbound, {n: len(a) for n, a in activities_by_dest.items()})
    nights = {b.destination: b.nights for b in stay_plan}
    sight_days = {b.destination: b.days - (1 if b.travel_day else 0) for b in stay_plan}
    rooms = max(1, math.ceil(spec.travellers / ROOM_SIZE))
    cities = [b.destination for b in stay_plan]

    hotel_rows = {c: world.hotels(region, c) for c in cities}
    accommodation_options = [
        Hotel(name=h["name"], destination=c, price_per_night_inr=h["price_per_night_inr"], rating=h["rating"], tier=h["tier"])
        for c, rows in hotel_rows.items() for h in rows
    ]
    chosen: dict[str, dict] = {}
    user_picked: set[str] = set()
    for c, rows in hotel_rows.items():
        if rows:
            chosen[c], by_user = pick_hotel(rows, c, tier, locks)
            if by_user:
                user_picked.add(c)

    def hotel_total() -> float:
        return sum(chosen[c]["price_per_night_inr"] * nights.get(c, 0) * rooms for c in chosen)

    food = estimate_food_costs(region, spec.duration_days, spec.travellers, tier)
    between_cities = route.total_cost_inr if route else 0.0
    local = estimate_local_transport(spec.duration_days, spec.travellers)
    transport = between_cities + local

    # Price only the sights that can fit: pace x sightseeing days in each city.
    per_day = PACE_CAPS.get(locks.pace or spec.constraints.pace, 3)
    kept: dict[str, list] = {}
    for name, acts in activities_by_dest.items():
        usable = sorted((a for a in acts if not activity_is_excluded(a.name, locks)), key=lambda a: (-a.preference_score, a.id))
        kept[name] = usable[: max(1, per_day * max(1, sight_days.get(name, 1)))]

    def activities_total() -> float:
        return calculate_activity_costs([a.cost_inr for acts in kept.values() for a in acts], spec.travellers)

    def total() -> float:
        return hotel_total() + food["total_inr"] + transport + activities_total()

    dropped: list[str] = []
    dropped_ids: list[str] = []
    over_at_first = total() > ceiling
    if over_at_first:
        # 1. Cheaper hotels first (the least damage to the trip), biggest saving first.
        saving = {c: (chosen[c]["price_per_night_inr"] - min(h["price_per_night_inr"] for h in rows)) * nights.get(c, 0)
                  for c, rows in hotel_rows.items() if c in chosen and rows and c not in user_picked}
        for c in sorted(saving, key=lambda c: (-saving[c], c)):
            if total() <= ceiling:
                break
            chosen[c] = min(hotel_rows[c], key=lambda h: h["price_per_night_inr"])
        # 2. Then paid sights, least value per rupee first (fractional-knapsack greedy), keeping one per city.
        removable = sorted(((d, a) for d, acts in kept.items() for a in acts if a.cost_inr > 0), key=lambda p: (p[1].preference_score / p[1].cost_inr, p[1].id))
        for dest, act in removable:
            if total() <= ceiling:
                break
            if len(kept[dest]) > 1:
                kept[dest].remove(act)
                dropped.append(f"{act.name} ({dest})")
                dropped_ids.append(act.id)

    grand_total = total()
    check = validate_budget(grand_total, ceiling)
    tradeoffs: list[str] = []
    if dropped:
        tradeoffs.append(f"Dropped the paid activities with the least value per rupee to close the gap: {', '.join(dropped)}")
    if not check["within_budget"]:
        hotel_saving = sum((chosen[c]["price_per_night_inr"] - min(h["price_per_night_inr"] for h in rows)) * nights.get(c, 0) * rooms
                           for c, rows in hotel_rows.items() if c in chosen and rows)
        paid = [a.cost_inr for acts in kept.values() for a in acts if a.cost_inr > 0]
        tradeoffs.extend(generate_tradeoff_options(hotel_saving, max(paid) * spec.travellers if paid else 0.0))

    hotels_inr, activities_inr = hotel_total(), activities_total()
    breakdown = BudgetBreakdown(
        hotels_inr=hotels_inr,
        food_inr=food["total_inr"],
        activities_inr=activities_inr,
        transport_inr=transport,
        total_inr=grand_total,
        ceiling_inr=ceiling,
        over_budget_by_inr=check["over_by_inr"],
        selected_hotels=[
            Hotel(name=chosen[c]["name"], destination=c, price_per_night_inr=chosen[c]["price_per_night_inr"], rating=chosen[c]["rating"], tier=chosen[c]["tier"])
            for c in cities if c in chosen
        ],
        line_items=[
            BudgetLineItem(category="hotels", amount_inr=hotels_inr, notes=f"{sum(nights.values())} night(s), {rooms} room(s)"),
            BudgetLineItem(category="food", amount_inr=food["total_inr"], notes=f"₹{food['per_person_per_day_inr']:.0f} per person per day"),
            BudgetLineItem(category="activities", amount_inr=activities_inr),
            BudgetLineItem(category="transport", amount_inr=transport, notes=f"between cities ₹{between_cities:,.0f} + local ₹{local:,.0f}"),
        ],
        tradeoff_suggestions=tradeoffs,
    )

    status = "within budget" if check["within_budget"] else f"over by ₹{check['over_by_inr']:,.0f}"
    stays = ", ".join(f"{b.destination} {b.nights}n" for b in stay_plan)
    message = (
        f"Budget Agent: {tier} tier, total ₹{grand_total:,.0f} vs ceiling ₹{ceiling:,.0f} ({status}"
        f"{f', dropped {len(dropped)} activity(ies)' if dropped else ''}); {rooms} room(s), stays {stays}."
    )
    return Result(
        updates={
            "budget_breakdown": breakdown,
            "accommodation_options": accommodation_options,
            "excluded_activity_ids": dropped_ids,
            "stay_plan": stay_plan,
        },
        posts=[Post(MsgType.BUDGET_READY, "itinerary_architect", f"₹{grand_total:,.0f} of ₹{ceiling:,.0f} ({status})")],
        message=message,
        tools=["hotels", "estimate_food_costs", "estimate_local_transport", "validate_budget"],
        algorithms=["greedy repair: cheapest hotels, then least value-per-rupee sights"] if over_at_first else [],
        note=f"{tier} tier" + (" (one cheaper: the last plan did not fit)" if cheaper else "") + (", over the ceiling at first" if over_at_first else ""),
    )


class BudgetAgent(Agent):
    name = "budget_agent"
    label = "Budget Optimization"
    role = "Money: prices hotels, food, sights and transport against the ceiling and repairs an overrun"
    architecture = "utility-based (trades comfort for money by the cheapest damage per rupee saved)"
    peas = {
        "performance": "Total at or under the ceiling; the fewest and least valuable sights dropped; the traveller's hotel choices kept",
        "environment": "Hotel prices per tier, food and local-transport prices, fares from Mobility, the ceiling and any reported budget cut",
        "actuators": "Write the budget breakdown, the hotels, the stay plan and the sights dropped; send BUDGET_READY",
        "sensors": "The route and its fares, the candidate sights, the trip spec, the edit locks, reported budget cuts, the previous breakdown",
    }
    reads = ("trip_spec", "selected_destinations", "route", "candidate_activities", "edit_locks", "disruptions", "budget_breakdown")
    writes = ("budget_breakdown", "accommodation_options", "excluded_activity_ids", "stay_plan")

    def handle(self, msg: Message, view: dict) -> Result:
        return price_trip(view)
