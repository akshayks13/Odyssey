"""
Agent 5 — Itinerary Architect

Builds the day-by-day, hour-level schedule. The scheduling core is OR-Tools
VRPTW (`pywrapcp.RoutingModel`) per day: opening-hour time windows, Mapbox
travel matrix, activity service times, and soft lunch/dinner nodes.
"""
from __future__ import annotations

from algorithms.csp_solver import solve_day_schedule
from algorithms.optimizer import compute_score, infer_archetype
from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import Coordinates, Itinerary, ItineraryDay, ScheduledItem
from orchestration.state import TripState
from tools.foursquare_api import get_opening_hours
from tools.mapbox_api import travel_time_matrix
from tools.schedule_validator import check_schedule_conflicts, validate_time_windows

_PACE_CAPS = {"relaxed": 2, "moderate": 3, "packed": 4}
_PACE_HOURS = {
    "relaxed": (9.0, 18.0),
    "moderate": (8.0, 21.0),
    "packed": (7.0, 22.0),
}


def _allocate_days(n_stops: int, duration: int) -> list[int]:
    if n_stops <= 0:
        return []
    base = duration // n_stops
    extra = duration % n_stops
    return [max(1, base + (1 if i < extra else 0)) for i in range(n_stops)]


def _coords_of(act: dict, fallback: Coordinates | None) -> dict | None:
    raw = act.get("coordinates")
    if isinstance(raw, dict) and raw.get("lat") is not None:
        return {"lat": raw["lat"], "lng": raw["lng"]}
    if fallback:
        return {"lat": fallback.lat, "lng": fallback.lng}
    return None


def _matrix_for_day(depot: Coordinates | None, acts: list[dict]) -> list[list[int]] | None:
    if depot is None:
        return None
    points = [{"lat": depot.lat, "lng": depot.lng}]
    for act in acts:
        coord = _coords_of(act, depot)
        if coord is None:
            return None
        points.append(coord)
    result = travel_time_matrix.invoke({"points": points})
    matrix = result.get("matrix")
    if not matrix or len(matrix) != len(points):
        return None
    return matrix


def itinerary_architect_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    budget = state.get("budget_breakdown")
    selected = state.get("selected_destinations", [])
    excluded = set(state.get("excluded_activity_ids") or [])
    dest_coords = {d.name: d.coordinates for d in selected}

    order = (route.ordered_destinations if route and route.ordered_destinations else [d.name for d in selected]) or [
        spec.destination_region
    ]

    days_alloc = _allocate_days(len(order), spec.duration_days)
    pace = spec.constraints.pace if spec.constraints.pace in _PACE_CAPS else "moderate"
    llm_note = ""
    decision = llm_decide(
        get_llm(),
        tools=[get_opening_hours, validate_time_windows, travel_time_matrix],
        system=(
            "You are Odyssey's Itinerary Architect. Call get_opening_hours and "
            "travel_time_matrix to inspect feasibility. Reply ONLY with JSON: "
            "{\"pace\": \"relaxed|moderate|packed\", \"reasoning\": \"...\"}."
        ),
        user=(
            f"Duration {spec.duration_days} days, ordered cities {order}, "
            f"user pace {spec.constraints.pace}, activities per city "
            f"{ {k: [a.name for a in v] for k, v in activities_by_dest.items()} }."
        ),
    )
    if decision.get("pace") in _PACE_CAPS:
        pace = decision["pace"]
        llm_note = f" LLM pace={pace}"
        if decision.get("_tool_calls"):
            llm_note += f" tools={decision['_tool_calls']}"
    elif decision.get("pace"):
        llm_note = f" LLM pace={decision['pace']}"
        if decision.get("_tool_calls"):
            llm_note += f" tools={decision['_tool_calls']}"

    day_start, day_end = _PACE_HOURS[pace]
    per_day_cap = _PACE_CAPS[pace]
    leg_by_destination = {leg.destination: leg for leg in route.legs} if route else {}

    itinerary_days: list[ItineraryDay] = []
    day_number = 1
    preference_scores_used: list[float] = []
    quality_scores_used: list[float] = []

    for idx, dest_name in enumerate(order):
        acts_pool = [
            a.model_dump()
            for a in activities_by_dest.get(dest_name, [])
            if a.id not in excluded and not a.is_closed
        ]
        acts_pool.sort(key=lambda a: a.get("preference_score") or 0, reverse=True)
        n_days_here = days_alloc[idx] if idx < len(days_alloc) else 1
        depot = dest_coords.get(dest_name)

        for day_in_dest in range(n_days_here):
            if day_number > spec.duration_days:
                break

            remaining_days = max(1, n_days_here - day_in_dest)
            take = min(len(acts_pool), max(per_day_cap, (len(acts_pool) + remaining_days - 1) // remaining_days))
            day_candidates = acts_pool[:take]

            for act in day_candidates:
                hours = get_opening_hours.invoke({"activity_id": act["id"], "destination": dest_name})
                act["opening_hour"] = hours.get("opening_hour", act.get("opening_hour", 9))
                act["closing_hour"] = hours.get("closing_hour", act.get("closing_hour", 18))

            matrix = _matrix_for_day(depot, day_candidates)
            result = solve_day_schedule(
                day_candidates,
                day_start_hour=day_start,
                day_end_hour=day_end,
                travel_matrix_minutes=matrix,
                include_meals=True,
                max_time_in_seconds=0.35,
            )
            items = [
                ScheduledItem(
                    activity_id=s["id"],
                    activity_name=s["name"],
                    destination=dest_name,
                    start_hour=s["start_hour"],
                    end_hour=s["end_hour"],
                    category=s["category"],
                )
                for s in result["scheduled"]
            ]
            scheduled_full = [a for a in day_candidates if a["id"] in result["selected_ids"]]
            used = set(result["selected_ids"])
            # Spread leftover activities (including ones the solver dropped) to later days.
            acts_pool = [a for a in acts_pool if a["id"] not in used]

            travel_leg = leg_by_destination.get(dest_name) if day_in_dest == 0 and idx > 0 else None

            itinerary_days.append(
                ItineraryDay(day_number=day_number, destination=dest_name, items=items, travel_leg=travel_leg)
            )

            preference_scores_used.extend(a["preference_score"] for a in scheduled_full)
            quality_scores_used.extend(a["rating"] / 5.0 for a in scheduled_full)

            day_number += 1

    validation = check_schedule_conflicts.invoke({"days": [d.model_dump() for d in itinerary_days]})

    pref_avg = sum(preference_scores_used) / len(preference_scores_used) if preference_scores_used else 0.5
    quality_avg = sum(quality_scores_used) / len(quality_scores_used) if quality_scores_used else 0.5

    total_travel_hours = route.total_duration_hours if route else 0.0
    total_activity_hours = sum(item.end_hour - item.start_hour for d in itinerary_days for item in d.items)
    total_time = total_travel_hours + total_activity_hours
    travel_burden = (total_travel_hours / total_time) if total_time > 0 else 0.0
    route_efficiency = 1.0 - min(1.0, travel_burden)

    if budget and budget.ceiling_inr > 0:
        budget_efficiency = max(0.0, 1.0 - max(0.0, budget.total_inr - budget.ceiling_inr) / budget.ceiling_inr)
    else:
        budget_efficiency = 1.0

    constraint_violation_penalty = min(1.0, len(validation["issues"]) * 0.15)

    archetype = infer_archetype(spec.preferences.as_dict(), spec.budget_inr, spec.travellers)
    score_result = compute_score(
        preference_satisfaction=pref_avg,
        activity_quality=quality_avg,
        route_efficiency=route_efficiency,
        budget_efficiency=budget_efficiency,
        travel_burden=travel_burden,
        constraint_violations=constraint_violation_penalty,
        archetype=archetype,
    )

    total_cost = budget.total_inr if budget else 0.0

    itinerary = Itinerary(
        days=itinerary_days,
        total_cost_inr=total_cost,
        optimization_score=score_result["total"],
        score_breakdown=score_result["breakdown"],
    )

    source = f" via {llm_provider_name()} + OR-Tools VRPTW" if llm_note else " via OR-Tools VRPTW"
    message = (
        f"Itinerary Architect: built {len(itinerary_days)}-day schedule{source} "
        f"(time windows + travel matrix + meal breaks), score {score_result['total']:.2f} "
        f"({archetype} profile).{llm_note}"
    )

    return {
        "draft_itinerary": itinerary,
        "optimization_score": score_result["total"],
        "agent_messages": [message],
    }
