"""Itinerary Architect — the day-by-day, hour-by-hour schedule.

Role: scheduler. Packs sights and meals into each day and says where each night is spent. Does not pick
cities or change the order.

Each day is a CSP (algorithms/csp_solver.py) solved by backtracking with MRV and forward checking:
sights must fit their opening hours, lunch 12-14 and dinner 19-21, with travel time between them.

Rules: pace sets the day (relaxed 9-18 / 2 sights, moderate 8-21 / 3, packed 7-22 / 4). An arrival day
starts after the transfer. A day the monthly average calls rainy tries indoor sights first. What the
field reports says overrides the average: on a day of heavy rain the outdoor sights are dropped from that
day (they stay available for later days), and a sight reported closed that day is dropped too. The
traveller's free days, light days and pinned sights are honoured. Afterwards the multi-objective score is
computed and the activity cost is replaced by what was actually scheduled.

The full system solves each day as a CSP; the "greedy_schedule" strategy uses a first-fit timetable instead.
"""
from __future__ import annotations

from datetime import date, timedelta

from agents.base import Agent, Post, Result
from algorithms.csp_solver import solve_day_schedule
from algorithms.optimizer import compute_score, infer_archetype
from algorithms.planning import PACE_CAPS, PACE_HOURS, activity_is_excluded, plan_stays
from core.messages import Message, MsgType
from core.strategies import Strategy
from models.schemas import Coordinates, DayWeather, EditLocks, Hotel, Itinerary, ItineraryDay, ObservedFacts, RouteLeg, ScheduledItem, TransportMode
from tools import world
from tools.cost_calculator import reconcile_activity_costs
from tools.schedule_validator import check_schedule_conflicts
from tools.world import OUTDOOR_CATEGORIES as OUTDOOR

MIN_SIGHTSEEING_WINDOW = 2.0  # hours; with less than this left, the day is a travel day


def _hotel_for_destination(budget, dest_name: str) -> Hotel | None:
    """The hotel picked for this city. None if there isn't one: never another city's hotel."""
    return next((h for h in (budget.selected_hotels if budget else []) if h.destination == dest_name), None)


def _matrix_for_day(depot: Coordinates, acts: list[dict]) -> list[list[int]]:
    """Minutes between the hotel (the city centre) and every sight of the day."""
    points = [{"lat": depot.lat, "lng": depot.lng}] + [a["coordinates"] for a in acts]
    return world.travel_time_matrix(points)


def _day_weather(w: dict, heavy_rain: bool, checked: bool) -> DayWeather:
    """The day's weather: the monthly average, or what the field report says when the day has been checked."""
    condition = "Heavy rain (field report)" if heavy_rain else w["condition"]
    has_temps = w["tmin"] is not None and w["tmax"] is not None
    return DayWeather(
        summary=f"{condition}, {w['tmin']:.0f}–{w['tmax']:.0f}°C" if has_temps else condition,
        condition=condition,
        tmin=w["tmin"],
        tmax=w["tmax"],
        rain_mm=w["rain_mm"],
        rain_chance=w["rain_chance"],
        rainy=w["rainy"] or heavy_rain,
        source="field report" if checked else w["source"],
    )


def _transfer_buffer(leg: RouteLeg) -> float:
    """Hours lost around a transfer besides the ride itself (station, check-in)."""
    return 1.0 if leg.mode == TransportMode.RAIL else 0.5


def _pinned_day(act_name: str, locks: EditLocks) -> int | None:
    low = act_name.lower()
    for name, day in locks.pinned_activities.items():
        if name.lower() in low or low in name.lower():
            return day
    return None


def build_schedule(state: dict, strategy: Strategy) -> Result:
    """The whole scheduling procedure, from the blackboard view to the agent's result."""
    spec = state["trip_spec"]
    route = state.get("route")
    activities_by_dest = state.get("candidate_activities", {})
    budget = state.get("budget_breakdown")
    selected = state.get("selected_destinations", [])
    excluded = set(state.get("excluded_activity_ids") or [])
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    observed: ObservedFacts = state.get("observed") or ObservedFacts()
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
    if locks.day_start_hour is not None:
        day_start = max(day_start, float(locks.day_start_hour))
    per_day_cap = PACE_CAPS[pace]
    leg_by_destination = {leg.destination: leg for leg in route.legs} if route else {}
    start_date = date.fromisoformat(spec.start_date) if spec.start_date else None
    activity_costs = {a.id: a.cost_inr for acts in activities_by_dest.values() for a in acts}

    itinerary_days: list[ItineraryDay] = []
    day_number = 1
    preference_scores_used: list[float] = []
    quality_scores_used: list[float] = []
    solver_runs: list[dict] = []  # stats of every CSP solve, so the trace reports what really ran

    for block in stay_plan:
        dest_name = block.destination
        pool = [
            a.model_dump()
            for a in activities_by_dest.get(dest_name, [])
            if a.id not in excluded and not activity_is_excluded(a.name, locks)
        ]
        pool.sort(key=lambda a: a.get("preference_score") or 0, reverse=True)
        depot = dest_coords.get(dest_name)
        inbound = leg_by_destination.get(dest_name)
        block_start = (start_date + timedelta(days=day_number - 1)).isoformat() if start_date else ""
        weather_by_date = (
            {w["date"]: w for w in (world.weather_on(spec.destination_region, dest_name, date.fromisoformat(block_start) + timedelta(days=i)) for i in range(block.days))}
            if start_date else {}
        )

        for day_in_block in range(block.days):
            if day_number > spec.duration_days:
                break
            day_iso = (start_date + timedelta(days=day_number - 1)).isoformat() if start_date else None
            today_weather = weather_by_date.get(day_iso) if day_iso else None
            heavy_rain = bool(day_iso) and observed.is_heavy_rain(dest_name, day_iso)  # reported, not just expected
            first_here = day_in_block == 0
            is_last_day = day_number >= spec.duration_days
            win_start, win_end = day_start, day_end
            kind, note = "sightseeing", None

            if first_here and inbound:
                win_start = day_start + inbound.duration_hours + _transfer_buffer(inbound)
                note = inbound.summary

            if block.travel_day and first_here:
                kind, note = "travel", f"{note or 'Travel day'} — most of the day is on the road."
            elif win_end - win_start < MIN_SIGHTSEEING_WINDOW:
                kind, note = "travel", f"{note or 'Travel day'} — too little daylight left for sightseeing."
            elif day_number in locks.free_days:
                kind, note = "leisure", "Free day, as you asked."

            items: list[ScheduledItem] = []
            if kind == "sightseeing":
                cap = 1 if day_number in locks.light_days else per_day_cap
                # What the field reports rules out today: outdoor sights in heavy rain, and sights closed today.
                rained_out = [a for a in pool if heavy_rain and a["category"] in OUTDOOR]
                closed_today = [a for a in pool if day_iso and observed.is_closed(a["id"], day_iso)]
                available = [a for a in pool if a not in rained_out and a not in closed_today]
                if rained_out:
                    note = " · ".join(x for x in (note, "Heavy rain reported — outdoor sights skipped today.") if x)
                if closed_today:
                    note = " · ".join(x for x in (note, f"Closed today: {', '.join(a['name'] for a in closed_today)}.") if x)
                pinned_today = [a for a in available if _pinned_day(a["name"], locks) == day_number]
                waiting = [a for a in available if a not in pinned_today and _pinned_day(a["name"], locks) is None]
                # Only sights open long enough inside today's window (a 6-9am class cannot follow an 11am arrival).
                waiting = [a for a in waiting if min(a["closing_hour"], win_end) - max(a["opening_hour"], win_start) >= a["duration_minutes"] / 60]
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
                        method=strategy.scheduler,
                    )
                    solver_runs.append(result["stats"])
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

                if not acts_out and first_here and inbound:
                    kind, note = "travel", " · ".join(x for x in (note, "Arriving too late for any sight to fit.") if x)
                elif not acts_out:
                    kind, note = "leisure", " · ".join(x for x in (note, "Nothing left to schedule here — free time.") if x)

            itinerary_days.append(
                ItineraryDay(
                    day_number=day_number,
                    destination=dest_name,
                    date=(start_date + timedelta(days=day_number - 1)).isoformat() if start_date else None,
                    kind=kind,
                    note=note,
                    weather=_day_weather(today_weather, heavy_rain, checked=bool(day_iso) and (heavy_rain or observed.is_dry(dest_name, day_iso))) if today_weather else None,
                    items=items,
                    travel_leg=inbound if first_here else None,
                    overnight_hotel=None if is_last_day else _hotel_for_destination(budget, dest_name),
                )
            )
            day_number += 1

    validation = check_schedule_conflicts([d.model_dump() for d in itinerary_days])

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

    updates: dict = {"draft_itinerary": itinerary, "optimization_score": score_result["total"]}
    if budget:
        # Budget priced the sights before they were scheduled; the Architect replaces that estimate with what it really scheduled.
        reconciled = reconcile_activity_costs(budget, itinerary, activity_costs, spec.travellers)
        itinerary.total_cost_inr = reconciled.total_inr
        updates["budget_breakdown"] = reconciled

    sightseeing_days = sum(1 for d in itinerary_days if any(i.kind == "activity" for i in d.items))
    solved = len(solver_runs)  # travel and free days are never sent to the solver
    nodes = sum(r["nodes"] for r in solver_runs)
    backtracks = sum(r["backtracks"] for r in solver_runs)
    solver_name = "CSP backtracking + MRV + forward checking" if strategy.scheduler == "csp" else "greedy first-fit timetable"
    # The schedule goes to the field check when the strategy observes, straight to the Critic when it does not.
    reviewer = "environment" if strategy.observe else "critic_replanner"
    return Result(
        updates=updates,
        posts=[Post(MsgType.SCHEDULE_READY, reviewer, f"{len(itinerary_days)} days, {sightseeing_days} with sights, score {score_result['total']:.2f}")],
        message=(
            f"Itinerary Architect: built {len(itinerary_days)}-day schedule "
            f"({sightseeing_days} sightseeing day(s), meals and travel time included), "
            f"score {score_result['total']:.2f} ({archetype} profile)."
        ),
        tools=["weather_on", "travel_time_matrix", "check_schedule_conflicts"],
        algorithms=([f"{solver_name} ({solved} day{'s' if solved != 1 else ''}, {nodes} nodes, {backtracks} backtracks)"] if solved else []) + ["multi-objective scoring"],
        note=f"{pace} pace, {archetype} profile, score {score_result['total']:.2f}",
    )


class ItineraryArchitect(Agent):
    name = "itinerary_architect"
    label = "Itinerary Architect"
    role = "Scheduler: packs sights and meals into each day and says where each night is spent"
    architecture = "goal-based (finds a consistent timetable: a constraint satisfaction problem per day)"
    peas = {
        "performance": "Every sight inside its opening hours and the day's window; no overlaps once travel time is counted; meals in their windows; the traveller's free, light and pinned days kept; high score",
        "environment": "Opening hours, distances inside a city, the stay plan, the weather (expected and reported), sights reported closed",
        "actuators": "Write the day-by-day schedule and the reconciled cost of what it scheduled; send SCHEDULE_READY",
        "sensors": "The route, the sights and their hours, the budget, the stay plan, the edit locks, the field reports",
    }
    reads = ("trip_spec", "route", "candidate_activities", "budget_breakdown", "selected_destinations", "excluded_activity_ids", "edit_locks", "stay_plan", "observed")
    writes = ("draft_itinerary", "optimization_score", "budget_breakdown")

    def handle(self, msg: Message, view: dict) -> Result:
        return build_schedule(view, self.strategy)
