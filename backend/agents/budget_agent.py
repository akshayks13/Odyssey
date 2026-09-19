"""
Agent 3 — Budget Optimization

Runs AFTER Mobility so the transport cost from the chosen route is included
in the total. Prices hotels + food + activities + transport, validates
against the ceiling, and (LLM-guided) proposes trade-offs when over budget.
Also reacts to injected budget-cut disruptions.
"""
from __future__ import annotations

from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import BudgetBreakdown, BudgetLineItem, DisruptionType, Hotel
from orchestration.state import TripState
from tools.travel_market import search_hotel_offers, search_hotels
from tools.budget_validator import generate_tradeoff_options, validate_budget
from tools.cost_calculator import calculate_activity_costs, estimate_food_costs


def _budget_tier(per_person_inr: float) -> str:
    if per_person_inr < 8000:
        return "budget"
    if per_person_inr > 20000:
        return "premium"
    return "mid"


def _llm_tradeoff_explanation(over_by: float, suggestions: list[str]) -> str | None:
    llm = get_llm()
    if llm is None or not suggestions:
        return None
    try:
        prompt = (
            "The trip is over budget by INR "
            f"{over_by:.0f}. Given these deterministic options: {suggestions}. "
            "In one short sentence, recommend which option causes the least damage "
            "to the traveller's experience and why."
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", None)
    except Exception:
        return None


def budget_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    selected = state.get("selected_destinations", [])
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    disruptions = state.get("disruptions", [])

    # Apply any injected budget-cut disruption to the ceiling before validating.
    ceiling = spec.budget_inr
    for d in disruptions:
        if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr:
            ceiling = d.new_budget_inr

    per_person = ceiling / max(spec.travellers, 1)
    tier = _budget_tier(per_person)
    llm_note = ""
    prefer_cheapest = True
    drop_names: list[str] = []

    decision = llm_decide(
        get_llm(),
        tools=[search_hotels, search_hotel_offers, estimate_food_costs, validate_budget, generate_tradeoff_options],
        system=(
            "You are Odyssey's Budget agent. Call tools to inspect hotels, food cost and "
            "whether the current plan fits the ceiling. Reply ONLY with JSON: "
            "{\"tier\": \"budget|mid|premium\", \"prefer_cheapest_hotels\": true, "
            "\"drop_activities\": [\"Activity Name\", ...], \"reasoning\": \"...\"}. "
            "Only drop activities if over budget. Prefer cheaper hotels before dropping "
            "high-preference activities."
        ),
        user=(
            f"Ceiling INR {ceiling}, travellers {spec.travellers}, days {spec.duration_days}, "
            f"destinations {[d.name for d in selected]}, "
            f"transport cost {(route.total_cost_inr if route else 0)}, "
            f"heuristic tier {tier}."
        ),
    )
    if decision:
        if decision.get("tier") in {"budget", "mid", "premium"}:
            tier = decision["tier"]
        if decision.get("prefer_cheapest_hotels") is False:
            prefer_cheapest = False
        drop_names = [str(n) for n in decision.get("drop_activities") or []]
        llm_note = f" LLM tier={tier}"
        if decision.get("_tool_calls"):
            llm_note += f" tools={decision['_tool_calls']}"

    nights = max(0, spec.duration_days - 1)
    nights_per_dest = max(1, nights // max(1, len(selected))) if selected else 0

    selected_hotels: list[Hotel] = []
    hotel_total = 0.0
    all_hotel_options: list[dict] = []
    accommodation_options: list[Hotel] = []
    for dest in selected:
        hotels_here = search_hotels.invoke({"destination": dest.name, "budget_tier": tier})
        for h in hotels_here:
            h["destination"] = dest.name
        all_hotel_options.extend(hotels_here)
        for h in hotels_here:
            accommodation_options.append(
                Hotel(
                    name=h["name"],
                    destination=dest.name,
                    price_per_night_inr=h["price_per_night_inr"],
                    rating=h.get("rating", 4.0),
                    source=h.get("source", "seed"),
                )
            )
        if hotels_here:
            pick = min(hotels_here, key=lambda h: h["price_per_night_inr"]) if prefer_cheapest else max(
                hotels_here, key=lambda h: h["rating"]
            )
            selected_hotels.append(
                Hotel(
                    name=pick["name"],
                    destination=dest.name,
                    price_per_night_inr=pick["price_per_night_inr"],
                    rating=pick["rating"],
                    source=pick["source"],
                )
            )
            hotel_total += pick["price_per_night_inr"] * nights_per_dest

    food_result = estimate_food_costs.invoke(
        {"duration_days": spec.duration_days, "travellers": spec.travellers, "tier": tier}
    )
    transport_cost = route.total_cost_inr if route else 0.0
    fixed_costs = hotel_total + food_result["total_inr"] + transport_cost

    # Working copy of activities we can trim if over budget — this is the
    # "cheapest-damage repair" the Critic's conflict protocol expects Budget
    # to perform: drop the lowest-preference, highest-cost activities first,
    # never leaving a selected destination with zero activities.
    trimmed_activities = {name: list(acts) for name, acts in activities_by_dest.items()}
    dropped: list[str] = []
    dropped_ids: list[str] = []

    # Honour explicit LLM drop list first (cheapest-damage choices the model made).
    if drop_names:
        for dest, acts in list(trimmed_activities.items()):
            keep = []
            for act in acts:
                if act.name in drop_names and len(acts) > 1:
                    dropped.append(f"{act.name} ({dest})")
                    dropped_ids.append(act.id)
                else:
                    keep.append(act)
            trimmed_activities[dest] = keep or acts[:1]

    def _activity_total_inr() -> float:
        costs = [act.cost_inr for acts in trimmed_activities.values() for act in acts]
        return calculate_activity_costs.invoke({"activity_costs_inr": costs, "travellers": spec.travellers})[
            "total_inr"
        ]

    activities_total = _activity_total_inr()
    total = fixed_costs + activities_total
    validation = validate_budget.invoke({"total_inr": total, "ceiling_inr": ceiling})

    if not validation["within_budget"]:
        # Sort all (destination, activity) pairs: lowest preference score
        # first, then highest cost first — the "least valuable, most
        # expensive" items are dropped first to close the gap fastest.
        removable = [
            (dest, act)
            for dest, acts in trimmed_activities.items()
            for act in acts
        ]
        removable.sort(key=lambda pair: (pair[1].preference_score, -pair[1].cost_inr))

        for dest, act in removable:
            if total <= ceiling:
                break
            if len(trimmed_activities[dest]) <= 1:
                continue  # keep at least one activity per destination
            trimmed_activities[dest].remove(act)
            activities_total = _activity_total_inr()
            total = fixed_costs + activities_total
            dropped.append(f"{act.name} ({dest})")
            dropped_ids.append(act.id)

        validation = validate_budget.invoke({"total_inr": total, "ceiling_inr": ceiling})

    tradeoffs: list[str] = []
    if dropped:
        tradeoffs.append(f"Dropped lowest-value activities to close budget gap: {', '.join(dropped)}")
    if not validation["within_budget"]:
        priciest_hotel = max(all_hotel_options, key=lambda h: h["price_per_night_inr"], default=None)
        cheapest_hotel = min(all_hotel_options, key=lambda h: h["price_per_night_inr"], default=None)
        hotel_diff = (
            (priciest_hotel["price_per_night_inr"] - cheapest_hotel["price_per_night_inr"]) * nights_per_dest
            if priciest_hotel and cheapest_hotel
            else 0.0
        )
        remaining_costs = [act.cost_inr for acts in trimmed_activities.values() for act in acts]
        tradeoff_result = generate_tradeoff_options.invoke(
            {
                "over_by_inr": validation["over_by_inr"],
                "hotel_price_diff_inr": hotel_diff,
                "activity_cost_inr": min(remaining_costs) if remaining_costs else 0.0,
            }
        )
        tradeoffs.extend(tradeoff_result["suggestions"])
        explanation = _llm_tradeoff_explanation(validation["over_by_inr"], tradeoffs)
        if explanation:
            tradeoffs = [explanation, *tradeoffs]

    breakdown = BudgetBreakdown(
        hotels_inr=hotel_total,
        food_inr=food_result["total_inr"],
        activities_inr=activities_total,
        transport_inr=transport_cost,
        total_inr=total,
        ceiling_inr=ceiling,
        over_budget_by_inr=validation["over_by_inr"],
        selected_hotels=selected_hotels,
        line_items=[
            BudgetLineItem(category="hotels", amount_inr=hotel_total),
            BudgetLineItem(category="food", amount_inr=food_result["total_inr"]),
            BudgetLineItem(category="activities", amount_inr=activities_total),
            BudgetLineItem(category="transport", amount_inr=transport_cost),
        ],
        tradeoff_suggestions=tradeoffs,
    )

    status = "within budget" if validation["within_budget"] else f"over by ₹{validation['over_by_inr']:.0f}"
    dropped_note = f", dropped {len(dropped)} activity(ies)" if dropped else ""
    source = f" via {llm_provider_name()} tool-calling" if llm_note else " via cost tools"
    message = f"Budget Agent: total ₹{total:.0f} vs ceiling ₹{ceiling:.0f} ({status}{dropped_note}){source}.{llm_note}"

    return {
        "budget_breakdown": breakdown,
        "accommodation_options": accommodation_options,
        "excluded_activity_ids": dropped_ids,
        "agent_messages": [message],
    }
