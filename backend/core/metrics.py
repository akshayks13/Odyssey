"""Score a finished plan against what really happens in the field.

The Critic can only check what the environment has told it. This module knows the whole truth (`FieldWorld`) and
what the traveller reported, so it can say how much of the plan would really work:

  - a sight is lost if it is outdoors on a heavy-rain day, is shut that day, sits in a city reported closed or hit by a
    storm, or falls on a day that drives through a strike;
  - the value that is left is scaled down by the share the plan overshoots the budget the traveller now has
    (the original, or the cut one they reported).

It is the equivalent of "% forest saved": the same field and the same reported events are used for every strategy,
so their scores are directly comparable. (A strike the traveller reports without a date only slows the road, which
the Mobility agent prices in, so it is not counted as a loss here.)
"""
from __future__ import annotations

from core.team import Team
from models.schemas import DisruptionType, TransportMode
from tools.world import OUTDOOR_CATEGORIES


def evaluate(team: Team) -> dict:
    """What the plan is worth, measured against the true field."""
    bb = team.bb.data
    itinerary = bb["final_itinerary"] or bb["draft_itinerary"]
    stats = team.stats()
    if itinerary is None or bb["trip_spec"] is None:
        return {"planned": False, "valid": False, "value_planned": 0.0, "value_delivered": 0.0, "sights_planned": 0, "sights_done": 0,
                "sights_lost": 0, "lost_travel_days": 0, "over_budget_pct": 0.0, "travel_hours": 0.0, "score": 0.0, **stats}

    region = bb["trip_spec"].destination_region
    truth = team.truth
    closed_cities = {d.target for d in team.reported if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)}
    cuts = [d.new_budget_inr for d in team.reported if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
    wanted = {a.id: a.preference_score for acts in bb["candidate_activities"].values() for a in acts}
    planned = delivered = 0.0
    sights_planned = sights_done = lost_travel_days = 0

    for day in itinerary.days:
        leg = day.travel_leg
        strike_day = bool(day.date and leg and leg.mode == TransportMode.ROAD and (truth.strike(leg.origin, day.date) or truth.strike(leg.destination, day.date)))
        lost_travel_days += strike_day
        for item in day.items:
            if item.kind != "activity":
                continue
            value = wanted.get(item.activity_id, 0.5)
            planned += value
            sights_planned += 1
            rained_out = bool(day.date) and item.category in OUTDOOR_CATEGORIES and truth.heavy_rain(region, day.destination, day.date)
            shut = bool(day.date) and truth.sight_closed(item.activity_id, day.date)
            if strike_day or rained_out or shut or day.destination in closed_cities:
                continue
            delivered += value
            sights_done += 1

    budget = bb["budget_breakdown"]
    ceiling = min([bb["trip_spec"].budget_inr, *cuts])  # the budget the traveller now has, whatever the plan was priced against
    over = max(0.0, budget.total_inr - ceiling) / ceiling if budget and ceiling > 0 else 0.0
    delivered *= 1.0 - min(1.0, over)
    report = bb["validation_report"]
    return {
        "planned": True,
        "valid": bool(report and report.valid),
        "value_planned": round(planned, 3),
        "value_delivered": round(delivered, 3),
        "sights_planned": sights_planned,
        "sights_done": sights_done,
        "sights_lost": sights_planned - sights_done,
        "lost_travel_days": lost_travel_days,
        "over_budget_pct": round(100 * over, 2),
        "travel_hours": bb["route"].total_duration_hours if bb["route"] else 0.0,  # between cities, on the chosen visiting order
        "score": itinerary.optimization_score,
        **stats,
    }
