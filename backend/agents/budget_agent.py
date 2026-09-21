"""Budget — price hotels, food, activities, and transport against the ceiling.

Role: money optimizer. Runs after Mobility so transport cost is already known.
Does not reorder cities or build the hour schedule.

Decides: hotel tier (budget/mid/premium), which picks the hotel (cheapest, middle, or
best rated); which activities to drop if over budget.

Computes: the stay plan (days and nights per city, shared with the Architect so the
bill matches the schedule); hotels per room x nights; food; activities for the sights
that will fit; transport (the journeys Mobility priced, plus a flat daily allowance for autos and cabs
between the sights); `validate_budget`. Cuts keep at least one activity per city.

Tools the model can call: `search_hotels`, `estimate_food_costs`. The code uses
`validate_budget`, `generate_tradeoff_options`, `calculate_activity_costs` and `estimate_local_transport`.
"""
from __future__ import annotations

import math

from algorithms.planning import PACE_CAPS, activity_is_excluded, plan_stays
from llm import agent_trace, get_llm, llm_decide, llm_json
from models.schemas import BudgetBreakdown, BudgetLineItem, DisruptionType, EditLocks, Hotel
from orchestration.state import TripState
from tools import pmap
from tools.budget_validator import generate_tradeoff_options, validate_budget
from tools.cost_calculator import calculate_activity_costs, estimate_food_costs, estimate_local_transport
from tools.travel_market import search_hotels

ROOM_SIZE = 2  # travellers per room


def _budget_tier(per_person_inr: float) -> str:
    if per_person_inr < 8000:
        return "budget"
    if per_person_inr > 20000:
        return "premium"
    return "mid"


def _llm_tradeoff_explanation(over_by: float, suggestions: list[str]) -> str | None:
    if not suggestions:
        return None
    advice = llm_json(
        "You advise travellers on trimming a trip budget. Reply ONLY with JSON.",
        f"The trip is over budget by INR {over_by:.0f}. Options: {suggestions}. "
        'In one short sentence, say which causes the least damage to the experience and why. Schema: {"advice": str}',
    ).get("advice")
    return str(advice) if advice else None


def _pick_hotel(rows: list[dict], city: str, tier: str, locks: EditLocks) -> tuple[dict, bool]:
    """One hotel for a city: the tier decides (budget = cheapest, mid = middle, premium = best rated).
    The bool says the user chose it, so budget cuts leave it alone."""
    pref = locks.hotel_prefs.get(city.lower()) or locks.hotel_prefs.get("*")
    if pref == "cheapest":
        return min(rows, key=lambda h: h["price_per_night_inr"]), True
    if pref == "best":
        return max(rows, key=lambda h: (h.get("rating", 0), -h["price_per_night_inr"])), True
    named = next((h for h in rows if pref and pref.lower() in h["name"].lower()), None)
    if named:
        return named, True
    by_price = sorted(rows, key=lambda h: h["price_per_night_inr"])
    if tier == "budget":
        return by_price[0], False
    if tier == "premium":
        return max(rows, key=lambda h: (h.get("rating", 0), -h["price_per_night_inr"])), False
    return by_price[len(by_price) // 2], False


def budget_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    selected = state.get("selected_destinations", [])
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    disruptions = state.get("disruptions", [])
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    previous = state.get("budget_breakdown")

    # Apply budget-cut disruptions to the ceiling first.
    ceiling = spec.budget_inr
    for d in disruptions:
        if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr:
            ceiling = d.new_budget_inr

    tier = _budget_tier(ceiling / max(spec.travellers, 1))
    llm_note = ""
    drop_names: list[str] = []
    was_over = previous.over_budget_by_inr if previous is not None else 0.0
    sent_back = [d.reason for d in state.get("replan_directives", []) if d.target_agent == "budget_agent"]

    decision = llm_decide(
        get_llm(),
        tools=[search_hotels, estimate_food_costs],
        system=(
            "You are Odyssey's Budget agent — the money optimizer, not the scheduler. "
            "ROLE: fit hotels, food, activities, and transport under the ceiling. "
            "YOU DECIDE: hotel tier (budget = cheapest hotels, mid = middle, premium = best rated) "
            "and which activities to drop if still over budget. "
            "YOU MUST NOT: change city order, invent rupee totals (tools compute those), "
            "or build the day plan. "
            "Call search_hotels and estimate_food_costs to see real prices before choosing the tier. "
            "Only drop activities if over budget. Prefer cheaper hotels before dropping "
            "high-preference activities. "
            "Reply ONLY with JSON: "
            "{\"tier\": \"budget|mid|premium\", "
            "\"drop_activities\": [\"Activity Name\", ...], \"reasoning\": \"...\"}."
        ),
        user=(
            f"Ceiling INR {ceiling}, travellers {spec.travellers}, days {spec.duration_days}, "
            f"destinations {[d.name for d in selected]}, "
            f"between-city transport cost {(route.total_cost_inr if route else 0)} (local autos and cabs are added on top), "
            f"heuristic tier {tier}."
            + (f" The previous plan was over budget by INR {was_over:.0f}: choose a cheaper tier." if was_over > 0 else "")
            + (f" The traveller asked for these hotels: {locks.hotel_prefs}." if locks.hotel_prefs else "")
            + (f" The reviewer sent this back because: {'; '.join(sent_back)}." if sent_back else "")
        ),
    )
    if decision:
        if decision.get("tier") in {"budget", "mid", "premium"}:
            tier = decision["tier"]
        drop_names = [str(n) for n in decision.get("drop_activities") or []]
        llm_note = f" LLM tier={tier}"  # the tools it called arrive as `agent_meta` and show as chips

    # Days and nights per city: the Architect schedules from the same plan.
    order = route.ordered_destinations if route and route.ordered_destinations else [d.name for d in selected]
    inbound = {leg.destination: leg.duration_hours for leg in (route.legs if route else [])}
    counts = {name: len(acts) for name, acts in activities_by_dest.items()}
    stay_plan = plan_stays(order, spec.duration_days, inbound, counts)
    nights = {b.destination: b.nights for b in stay_plan}
    sight_days = {b.destination: b.days - (1 if b.travel_day else 0) for b in stay_plan}
    rooms = max(1, math.ceil(spec.travellers / ROOM_SIZE))

    cities = [b.destination for b in stay_plan if b.destination in {d.name for d in selected}]
    hotel_rows = pmap(lambda c: search_hotels.invoke({"destination": c, "budget_tier": tier}), cities, workers=4)
    hotels_by_city = {c: [dict(h) for h in (rows or [])] for c, rows in zip(cities, hotel_rows)}

    accommodation_options = [
        Hotel(name=h["name"], destination=c, price_per_night_inr=h["price_per_night_inr"], rating=h.get("rating", 4.0), source=h.get("source", "seed"))
        for c, rows in hotels_by_city.items()
        for h in rows
    ]
    chosen: dict[str, dict] = {}
    user_picked: set[str] = set()
    for city, rows in hotels_by_city.items():
        if rows:
            chosen[city], picked_by_user = _pick_hotel(rows, city, tier, locks)
            if picked_by_user:
                user_picked.add(city)

    def hotel_total() -> float:
        return sum(chosen[c]["price_per_night_inr"] * nights.get(c, 0) * rooms for c in chosen)

    food = estimate_food_costs.invoke(
        {"duration_days": spec.duration_days, "travellers": spec.travellers, "tier": tier, "region": spec.destination_region}
    )
    between_cities = route.total_cost_inr if route else 0.0
    local = estimate_local_transport.invoke({"duration_days": spec.duration_days, "travellers": spec.travellers})["total_inr"]
    transport_cost = between_cities + local  # the journeys, plus autos and cabs between the sights

    # Price the sights that will fit the trip (pace x days in each city), not every candidate.
    per_day = PACE_CAPS.get(locks.pace or spec.constraints.pace, 3)
    kept: dict[str, list] = {}
    for name, acts in activities_by_dest.items():
        usable = sorted((a for a in acts if not activity_is_excluded(a.name, locks)), key=lambda a: a.preference_score, reverse=True)
        kept[name] = usable[: max(1, per_day * max(1, sight_days.get(name, 1)))]
    dropped: list[str] = []
    dropped_ids: list[str] = []

    for dest, acts in kept.items():
        for act in list(acts):
            if act.name in drop_names and len(acts) > 1:
                acts.remove(act)
                dropped.append(f"{act.name} ({dest})")
                dropped_ids.append(act.id)

    def activities_total() -> float:
        costs = [a.cost_inr for acts in kept.values() for a in acts]
        return calculate_activity_costs.invoke({"activity_costs_inr": costs, "travellers": spec.travellers})["total_inr"]

    def total() -> float:
        return hotel_total() + food["total_inr"] + transport_cost + activities_total()

    # Over budget: drop the lowest-value paid activities (only if that could close the gap), then take
    # the cheapest hotel in cities the traveller did not choose a hotel for.
    over = total() - ceiling
    if over > 0:
        removable = sorted(
            ((d, a) for d, acts in kept.items() for a in acts if a.cost_inr > 0),
            key=lambda pair: (pair[1].preference_score, -pair[1].cost_inr),
        )
        can_save = sum(a.cost_inr for d, a in removable if len(kept[d]) > 1) * spec.travellers
        if can_save >= over:
            for dest, act in removable:
                if total() <= ceiling:
                    break
                if len(kept[dest]) > 1:
                    kept[dest].remove(act)
                    dropped.append(f"{act.name} ({dest})")
                    dropped_ids.append(act.id)
        for city, rows in hotels_by_city.items():
            if total() > ceiling and city not in user_picked and rows:
                chosen[city] = min(rows, key=lambda h: h["price_per_night_inr"])

    grand_total = total()
    validation = validate_budget.invoke({"total_inr": grand_total, "ceiling_inr": ceiling})

    tradeoffs: list[str] = []
    if dropped:
        tradeoffs.append(f"Dropped lowest-value activities to close budget gap: {', '.join(dropped)}")
    if not validation["within_budget"]:
        all_rows = [h for rows in hotels_by_city.values() for h in rows]
        avg_nights = max(1, round(sum(nights.values()) / max(1, len(nights))))
        hotel_diff = (
            (max(h["price_per_night_inr"] for h in all_rows) - min(h["price_per_night_inr"] for h in all_rows)) * avg_nights * rooms
            if all_rows
            else 0.0
        )
        remaining_costs = [a.cost_inr for acts in kept.values() for a in acts if a.cost_inr > 0]
        options = generate_tradeoff_options.invoke(
            {
                "over_by_inr": validation["over_by_inr"],
                "hotel_price_diff_inr": hotel_diff,
                "activity_cost_inr": min(remaining_costs) if remaining_costs else 0.0,
            }
        )
        tradeoffs.extend(options["suggestions"])
        explanation = _llm_tradeoff_explanation(validation["over_by_inr"], tradeoffs)
        if explanation:
            tradeoffs = [explanation, *tradeoffs]

    hotels_inr, activities_inr = hotel_total(), activities_total()
    breakdown = BudgetBreakdown(
        hotels_inr=hotels_inr,
        food_inr=food["total_inr"],
        activities_inr=activities_inr,
        transport_inr=transport_cost,
        total_inr=grand_total,
        ceiling_inr=ceiling,
        over_budget_by_inr=validation["over_by_inr"],
        selected_hotels=[
            Hotel(name=chosen[c]["name"], destination=c, price_per_night_inr=chosen[c]["price_per_night_inr"], rating=chosen[c].get("rating", 4.0), source=chosen[c].get("source", "seed"))
            for c in cities
            if c in chosen
        ],
        line_items=[
            BudgetLineItem(category="hotels", amount_inr=hotels_inr, notes=f"{sum(nights.values())} night(s), {rooms} room(s)"),
            BudgetLineItem(category="food", amount_inr=food["total_inr"]),
            BudgetLineItem(category="activities", amount_inr=activities_inr),
            BudgetLineItem(category="transport", amount_inr=transport_cost, notes=f"between cities ₹{between_cities:,.0f} + local ₹{local:,.0f}"),
        ],
        tradeoff_suggestions=tradeoffs,
    )

    status = "within budget" if validation["within_budget"] else f"over by ₹{validation['over_by_inr']:.0f}"
    dropped_note = f", dropped {len(dropped)} activity(ies)" if dropped else ""
    stays = ", ".join(f"{b.destination} {b.nights}n" for b in stay_plan)
    message = f"Budget Agent: total ₹{grand_total:.0f} vs ceiling ₹{ceiling:.0f} ({status}{dropped_note}); {rooms} room(s), stays {stays}.{llm_note}"

    return {
        "budget_breakdown": breakdown,
        "accommodation_options": accommodation_options,
        "excluded_activity_ids": dropped_ids,
        "stay_plan": stay_plan,
        # Trimming is a greedy heuristic and only runs when the plan starts over the ceiling; a plan that
        # already fits was never trimmed, so it must not claim to have been.
        "agent_meta": agent_trace(
            decision,
            algorithms=["greedy trimming: lowest-value sights first, then cheapest hotels"] if over > 0 else [],
            note=f"over the ceiling at first; dropped {len(dropped)} sight(s)" if over > 0 and dropped else ("over the ceiling at first" if over > 0 else ""),
        ),
        "agent_messages": [message],
    }
