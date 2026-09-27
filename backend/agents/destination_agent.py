"""Destination Discovery — choose which cities to visit and load their sights.

Role: explorer. Ranks the region's cities for this traveller; does not order them or book hotels.

Rules:
  1. Score every city: cosine similarity between the traveller's interests and the city's profile.
     A city where rain is likely on most trip days loses 10%.
  2. Never pick a city that is closed, hit by a storm, excluded by the traveller, or that the Critic
     said to avoid.
  3. Cities the traveller named (or pinned in an edit) come first.
  4. Fill up to ceil(days / 2) cities (max 4), best score first, but only cities within one day's
     travel (UCS road time <= the daily limit) of a city already chosen, so the stops stay close. Without
     named cities the set is grown from every seed city and the best total score wins.
  5. Sent back by the Critic: keep the chosen cities that are still fine, replace only the blocked one;
     for empty days (too few sights) take one city more.
  6. Each sight gets a preference score: the traveller's weight for its category (80%) and its rating (20%).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from agents.common import trace
from algorithms.planning import activity_is_excluded
from models.schemas import Activity, Coordinates, Destination, DisruptionType, EditLocks, TripSpec
from orchestration.state import TripState
from tools import world
from tools.preference_scorer import score_preference_match, sight_preference

RAIN_PENALTY = 0.9
DISRUPTED_PENALTY = 0.15


def _trip_dates(spec: TripSpec) -> list[date]:
    start = date.fromisoformat(spec.start_date) if spec.start_date else date.today()
    return [start + timedelta(days=i) for i in range(spec.duration_days)]


def rank_cities(spec: TripSpec, disrupted: set[str]) -> list[Destination]:
    prefs = spec.preferences.as_dict()
    dates = _trip_dates(spec)
    ranked = []
    for c in world.cities(spec.destination_region):
        score = score_preference_match(prefs, c["profile"])
        days = [world.weather_on(spec.destination_region, c["name"], d) for d in dates]
        rainy = sum(1 for d in days if d["rainy"])
        risky = rainy >= max(1, round(0.5 * len(days)))
        if risky:
            score *= RAIN_PENALTY
        if c["name"] in disrupted:
            score *= DISRUPTED_PENALTY
        first = days[0]
        ranked.append(
            Destination(
                name=c["name"],
                region=spec.destination_region,
                coordinates=Coordinates(**c["coordinates"]),
                preference_score=round(score, 4),
                description=c["description"],
                tags=c["tags"],
                weather_summary=f"{first['condition']}, {first['tmin']:.0f}–{first['tmax']:.0f}°C, rain likely on {rainy} of {len(days)} days",
                weather_risk=risky or c["name"] in disrupted,
            )
        )
    ranked.sort(key=lambda d: (-d.preference_score, d.name))
    return ranked


def choose_cities(spec: TripSpec, ranked: list[Destination], blocked: set[str], wanted: list[str], limit: int) -> list[Destination]:
    """The best set of up to `limit` cities, each within a day's travel of another in the set.

    With cities the traveller named (or kept from the last plan) the set grows from those. Otherwise it is
    grown from every possible seed city and the set with the highest total score wins, so one remote
    top-scoring city cannot leave the trip with a single stop.
    """
    region = spec.destination_region
    cap = spec.constraints.max_daily_travel_hours
    viable = [d for d in ranked if d.name not in blocked]
    by_name = {d.name: d for d in viable}

    def grow(chosen: list[Destination]) -> list[Destination]:
        chosen = list(chosen)
        for d in viable:
            if len(chosen) >= limit:
                break
            if d not in chosen and any(world.road_route(region, c.name, d.name)["hours"] <= cap for c in chosen):
                chosen.append(d)
        return chosen

    base = [by_name[n] for n in dict.fromkeys(wanted) if n in by_name]
    if base:
        return grow(base[:limit])
    clusters = [grow([seed]) for seed in viable]
    return max(clusters, key=lambda c: (len(c), sum(d.preference_score for d in c)), default=[])


def load_sights(spec: TripSpec, city_name: str, closed: set[str], locks: EditLocks) -> list[Activity]:
    prefs = spec.preferences.as_dict()
    out = []
    for s in world.sights(spec.destination_region, city_name):
        if s["name"] in closed or s["id"] in closed or activity_is_excluded(s["name"], locks):
            continue
        pref = 0.8 * sight_preference(prefs, s["category"]) + 0.2 * (s["rating"] / 5.0)
        out.append(
            Activity(
                id=s["id"],
                name=s["name"],
                destination=city_name,
                category=s["category"],
                duration_minutes=s["duration_minutes"],
                cost_inr=s["cost_inr"],
                rating=s["rating"],
                opening_hour=s["opening_hour"],
                closing_hour=s["closing_hour"],
                preference_score=round(pref, 4),
                coordinates=Coordinates(**s["coordinates"]),
                source="dataset",
            )
        )
    return out


def destination_agent_node(state: TripState) -> dict:
    spec: TripSpec = state["trip_spec"]
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    disruptions = state.get("disruptions", [])
    region = spec.destination_region

    disrupted = {d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)}
    avoid = {c for d in state.get("replan_directives", []) if d.target_agent == "destination_agent" for c in d.constraints.get("avoid", [])}
    excluded = {world.match_city(region, c) or c for c in locks.excluded_cities}
    blocked = disrupted | avoid | excluded

    ranked = rank_cities(spec, disrupted)
    named = [world.match_city(region, c) or c for c in locks.pinned_cities] or world.cities_named_in(region, spec.raw_input)
    wanted = list(named)
    usual = min(4, spec.constraints.max_destinations, math.ceil(spec.duration_days / 2))

    # Sent back by the Critic: keep the cities that are still fine and replace only what was blocked.
    # Days with nothing to see mean too few sights, so take one more city.
    sent_back = [d for d in state.get("replan_directives", []) if d.target_agent == "destination_agent"]
    if sent_back:
        wanted = wanted + [d.name for d in state.get("selected_destinations", []) if d.name not in blocked]
        if sent_back[-1].constraints.get("issue") == "LEISURE_DAY":
            usual += 1
    wanted = [c for c in dict.fromkeys(wanted) if c not in blocked]
    limit = max(1, min(spec.duration_days, max(usual, len(wanted))))  # everything wanted, but never more cities than days
    selected = choose_cities(spec, ranked, blocked, wanted, limit)
    activities = {d.name: load_sights(spec, d.name, disrupted, locks) for d in selected}

    notes = []
    if named:
        notes.append(f"you asked for {', '.join(named)}")
    kept = [c for c in wanted if c not in named]
    if kept:
        notes.append(f"kept {', '.join(kept)}")
    if blocked:
        notes.append(f"avoided {', '.join(sorted(blocked))}")
    message = (
        f"Destination Agent: ranked {len(ranked)} cities by interest match, selected "
        f"{', '.join(f'{d.name} ({d.preference_score:.2f})' for d in selected)}"
        + (f"; {'; '.join(notes)}" if notes else "") + "."
    )
    return {
        "candidate_destinations": ranked,
        "selected_destinations": selected,
        "candidate_activities": activities,
        "excluded_activity_ids": [],
        "agent_meta": trace(
            tools=["score_preference_match", "weather_on", "road_route", "sights"],
            algorithms=["cosine preference ranking", "UCS road-time proximity check"],
            note=f"up to {limit} cities for {spec.duration_days} days, within {spec.constraints.max_daily_travel_hours:g}h of each other"
            + ("; kept the cities still fine and replaced the rest" if sent_back else ""),
        ),
        "agent_messages": [message],
    }
