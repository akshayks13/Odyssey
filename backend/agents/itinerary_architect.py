"""Itinerary Architect — day-by-day hour schedule with OR-Tools VRPTW.

Role: scheduler. Packs activities, meals, and overnight hotels into days.
Does not pick new cities or change the A* order.

Decides: nothing by model. It is the solver agent: the pace (relaxed / moderate / packed) comes from the
Analyst's reading of the request, and sets daily start/end and how many activities to attempt.
Honours the user's pins, free days and light days.

Computes: weather for each day (a rainy day schedules indoor sights first); days per city (the same stay plan Budget priced from); time lost to
arriving, long transfers and the trip home; opening hours; Mapbox travel matrix;
OR-Tools VRPTW with meals; dates; multi-objective score. Replaces Budget's
activity estimate with what was actually scheduled.

No model call. The code uses `travel_time_matrix`, `check_schedule_conflicts` and the OR-Tools solver.
"""
from __future__ import annotations

from datetime import date, timedelta

from algorithms.csp_solver import solve_day_schedule
from llm import agent_trace
from algorithms.optimizer import compute_score, infer_archetype
from algorithms.planning import PACE_CAPS, PACE_HOURS, activity_is_excluded, plan_stays
from models.schemas import Coordinates, DayWeather, EditLocks, Hotel, Itinerary, ItineraryDay, RouteLeg, ScheduledItem, TransportMode
from orchestration.state import TripState
from tools.cost_calculator import reconcile_activity_costs
from tools.mapbox_api import travel_time_matrix
from tools.schedule_validator import check_schedule_conflicts
from tools.weather import trip_weather

OUTDOOR = {"nature", "adventure"}  # what rain spoils
MIN_SIGHTSEEING_WINDOW = 2.0  # hours; with less than this left, the day is a travel day


def _coords_of(act: dict, fallback: Coordinates | None) -> dict | None:
    raw = act.get("coordinates")
    if isinstance(raw, dict) and raw.get("lat") is not None:
        return {"lat": raw["lat"], "lng": raw["lng"]}
    if fallback:
        return {"lat": fallback.lat, "lng": fallback.lng}
    return None


def _hotel_for_destination(budget, dest_name: str) -> Hotel | None:
    """The hotel picked for this city. None if there isn't one: never another city's hotel."""
    return next((h for h in (budget.selected_hotels if budget else []) if h.destination == dest_name), None)


def _matrix_for_day(depot: Coordinates | None, acts: list[dict]) -> list[list[int]] | None:
    if depot is None:
        return None
    points = [{"lat": depot.lat, "lng": depot.lng}]
    for act in acts:
        coord = _coords_of(act, depot)
        if coord is None:
            return None
        points.append(coord)
    matrix = travel_time_matrix.invoke({"points": points}).get("matrix")
    return matrix if matrix and len(matrix) == len(points) else None


def _transfer_buffer(leg: RouteLeg) -> float:
    """Hours lost around a transfer besides the ride itself (airport or station, check-in)."""
    return 1.5 if leg.mode in (TransportMode.AIR, TransportMode.RAIL) else 0.5


def _pinned_day(act_name: str, locks: EditLocks) -> int | None:
    low = act_name.lower()
    for name, day in locks.pinned_activities.items():
        if name.lower() in low or low in name.lower():
            return day
    return None


def itinerary_architect_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    budget = state.get("budget_breakdown")
    selected = state.get("selected_destinations", [])
    excluded = set(state.get("excluded_activity_ids") or [])
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    dest_coords = {d.name: d.coordinates for d in selected}

    order = (route.ordered_destinations if route and route.ordered_destinations else [d.name for d in selected]) or [
        spec.destination_region
    ]
    stay_plan = state.get("stay_plan") or plan_stays(
        order,
        spec.duration_days,
        {leg.destination: leg.duration_hours for leg in (route.legs if route else [])},
        {name: len(acts) for name, acts in activities_by_dest.items()},
    )

    pace = locks.pace or (spec.constraints.pace if spec.constraints.pace in PACE_CAPS else "moderate")  # the Analyst read it from the request

    day_start, day_end = PACE_HOURS[pace]
    arrival_start = day_start  # "start later" is about mornings on the ground, not when the flight lands
    if locks.day_start_hour is not None:
        day_start = max(day_start, float(locks.day_start_hour))
    per_day_cap = PACE_CAPS[pace]
    leg_by_destination = {leg.destination: leg for leg in route.legs} if route else {}
    return_leg = route.return_leg if route else None
    start_date = date.fromisoformat(spec.start_date) if spec.start_date else None
    activity_costs = {a.id: a.cost_inr for acts in activities_by_dest.values() for a in acts}

    itinerary_days: list[ItineraryDay] = []
    day_number = 1
    preference_scores_used: list[float] = []
    quality_scores_used: list[float] = []
    solver_runs: list[str] = []  # the status of every OR-Tools solve, so the trace reports what really ran

    for block in stay_plan:
        dest_name = block.destination
        pool = [
            a.model_dump()
            for a in activities_by_dest.get(dest_name, [])
            if a.id not in excluded and not a.is_closed and not activity_is_excluded(a.name, locks)
        ]
        pool.sort(key=lambda a: a.get("preference_score") or 0, reverse=True)
        depot = dest_coords.get(dest_name)
        inbound = leg_by_destination.get(dest_name)
        block_start = (start_date + timedelta(days=day_number - 1)).isoformat() if start_date else ""
        weather_by_date = {r["date"]: r for r in trip_weather(dest_name, block_start, block.days)["days"]} if start_date else {}

        for day_in_block in range(block.days):
            if day_number > spec.duration_days:
                break
            today_weather = weather_by_date.get((start_date + timedelta(days=day_number - 1)).isoformat()) if start_date else None
            first_here = day_in_block == 0
            is_last_day = day_number >= spec.duration_days
            win_start, win_end = day_start, day_end
            kind, note = "sightseeing", None

            if first_here and inbound:
                depart = arrival_start if day_number == 1 else day_start  # only the trip's own arrival has a fixed time
                win_start = depart + inbound.duration_hours + _transfer_buffer(inbound)
                note = inbound.summary
            if is_last_day and return_leg:
                win_end = day_end - return_leg.duration_hours - _transfer_buffer(return_leg)
                note = " · ".join(x for x in (note, return_leg.summary) if x)

            if block.travel_day and first_here:
                kind, note = "travel", f"{note or 'Travel day'} — most of the day is on the road."
            elif win_end - win_start < MIN_SIGHTSEEING_WINDOW:
                kind, note = "travel", f"{note or 'Travel day'} — too little daylight left for sightseeing."
            elif day_number in locks.free_days:
                kind, note = "leisure", "Free day, as you asked."

            items: list[ScheduledItem] = []
            if kind == "sightseeing":
                cap = 1 if day_number in locks.light_days else per_day_cap
                pinned_today = [a for a in pool if _pinned_day(a["name"], locks) == day_number]
                waiting = [a for a in pool if a not in pinned_today and _pinned_day(a["name"], locks) is None]
                if today_weather and today_weather["rainy"]:  # rain: indoor sights first, outdoor ones wait for a dry day
                    waiting.sort(key=lambda a: (a.get("preference_score") or 0) * (0.4 if a.get("category") in OUTDOOR else 1.0), reverse=True)
                    note = " · ".join(x for x in (note, "Rain likely — indoor sights first.") if x)

                def solve(candidates: list[dict]) -> dict:
                    result = solve_day_schedule(
                        candidates,
                        day_start_hour=win_start,
                        day_end_hour=win_end,
                        travel_matrix_minutes=_matrix_for_day(depot, candidates),
                        include_meals=True,
                        max_time_in_seconds=0.35,
                    )
                    solver_runs.append(result["status"])
                    return result

                candidates = pinned_today + waiting[: max(0, cap - len(pinned_today))]
                result = solve(candidates)
                for i in range(cap, min(len(waiting), 3 * cap + 3), cap):  # none fit the day's hours: the next best, still within the cap
                    if result["scheduled"]:
                        break
                    candidates = pinned_today + waiting[i : i + cap]
                    result = solve(candidates)

                by_id = {a["id"]: a for a in candidates}
                acts_out = [
                    ScheduledItem(
                        activity_id=s["id"],
                        activity_name=s["name"],
                        destination=dest_name,
                        start_hour=s["start_hour"],
                        end_hour=s["end_hour"],
                        category=s["category"],
                        cost_inr=by_id[s["id"]].get("cost_inr", 0.0),
                    )
                    for s in result["scheduled"]
                ]
                meals_out = [
                    ScheduledItem(
                        activity_id=m["id"],
                        activity_name=m["name"],
                        destination=dest_name,
                        start_hour=m["start_hour"],
                        end_hour=m["end_hour"],
                        category="meal",
                        kind="meal",
                    )
                    for m in result["meals"]
                ]
                items = sorted([*acts_out, *meals_out], key=lambda i: i.start_hour)

                used = set(result["selected_ids"])
                scheduled_full = [a for a in candidates if a["id"] in used]
                pool = [a for a in pool if a["id"] not in used]
                preference_scores_used.extend(a["preference_score"] for a in scheduled_full)
                quality_scores_used.extend(a["rating"] / 5.0 for a in scheduled_full)

                if not acts_out:
                    kind, note = "leisure", " · ".join(x for x in (note, "Nothing left to schedule here — free time.") if x)

            itinerary_days.append(
                ItineraryDay(
                    day_number=day_number,
                    destination=dest_name,
                    date=(start_date + timedelta(days=day_number - 1)).isoformat() if start_date else None,
                    kind=kind,
                    note=note,
                    weather=(
                        DayWeather(
                            summary=f"{today_weather['condition']}, {today_weather['tmin']:.0f}–{today_weather['tmax']:.0f}°C"
                            if today_weather["tmin"] is not None and today_weather["tmax"] is not None else today_weather["condition"],
                            condition=today_weather["condition"],
                            tmin=today_weather["tmin"],
                            tmax=today_weather["tmax"],
                            rain_mm=today_weather["rain_mm"],
                            rain_chance=today_weather["rain_chance"],
                            rainy=today_weather["rainy"],
                            source=today_weather["source"],
                        )
                        if today_weather
                        else None
                    ),
                    items=items,
                    travel_leg=inbound if first_here else None,
                    departure_leg=return_leg if is_last_day else None,
                    overnight_hotel=None if is_last_day else _hotel_for_destination(budget, dest_name),
                )
            )
            day_number += 1

    validation = check_schedule_conflicts.invoke({"days": [d.model_dump() for d in itinerary_days]})

    pref_avg = sum(preference_scores_used) / len(preference_scores_used) if preference_scores_used else 0.5
    quality_avg = sum(quality_scores_used) / len(quality_scores_used) if quality_scores_used else 0.5

    total_travel_hours = route.total_duration_hours if route else 0.0
    total_activity_hours = sum(i.end_hour - i.start_hour for d in itinerary_days for i in d.items if i.kind == "activity")
    total_time = total_travel_hours + total_activity_hours
    travel_burden = (total_travel_hours / total_time) if total_time > 0 else 0.0
    route_efficiency = 1.0 - min(1.0, travel_burden)

    if budget and budget.ceiling_inr > 0:
        budget_efficiency = max(0.0, 1.0 - max(0.0, budget.total_inr - budget.ceiling_inr) / budget.ceiling_inr)
    else:
        budget_efficiency = 1.0

    archetype = infer_archetype(spec.preferences.as_dict(), spec.budget_inr, spec.travellers)
    score_result = compute_score(
        preference_satisfaction=pref_avg,
        activity_quality=quality_avg,
        route_efficiency=route_efficiency,
        budget_efficiency=budget_efficiency,
        travel_burden=travel_burden,
        constraint_violations=min(1.0, len(validation["issues"]) * 0.15),
        archetype=archetype,
    )

    itinerary = Itinerary(
        days=itinerary_days,
        total_cost_inr=budget.total_inr if budget else 0.0,
        optimization_score=score_result["total"],
        score_breakdown=score_result["breakdown"],
    )

    out: dict = {"draft_itinerary": itinerary, "optimization_score": score_result["total"]}
    if budget:
        reconciled = reconcile_activity_costs(budget, itinerary, activity_costs, spec.travellers)
        itinerary.total_cost_inr = reconciled.total_inr
        out["budget_breakdown"] = reconciled

    sightseeing_days = sum(1 for d in itinerary_days if any(i.kind == "activity" for i in d.items))
    # Travel and free days are never sent to the solver, so a plan made only of those did not use OR-Tools at all.
    solved = len(solver_runs)
    out["agent_meta"] = agent_trace(
        algorithms=(
            [f"OR-Tools routing solver: VRPTW, guided local search ({solved} day{'s' if solved != 1 else ''} solved)"] if solved else []
        ) + ["multi-objective scoring"],
        note=f"{pace} pace, {archetype} profile, score {score_result['total']:.2f}",
    )
    out["agent_messages"] = [
        f"Itinerary Architect: built {len(itinerary_days)}-day schedule "
        f"({sightseeing_days} sightseeing day(s), meals and travel time included), "
        f"score {score_result['total']:.2f} ({archetype} profile){' via OR-Tools VRPTW' if solved else ''}."
    ]
    return out
