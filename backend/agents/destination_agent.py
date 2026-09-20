"""Destination Discovery — pick which cities to keep and load their sights.

Role: explorer. Rank candidate cities for the parsed TripSpec; do not order
them on a map and do not book hotels.

Decides: which 1..N cities to keep (never a closed/weather-risky city; never
the origin/home city as a sightseeing stop).

Computes: weighted cosine preference score per city; weather risk; attraction
list per selected city.

Uses: `search_destinations` (seed region or Mapbox geocode), `search_attractions`
(LLM sights, else Foursquare nearby, else seed), `get_weather_forecast`,
`score_preference_match`, `get_place_photos`.
"""
from __future__ import annotations

import re

from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import Activity, Coordinates, Destination, DisruptionType, TripSpec
from orchestration.state import TripState
from tools.foursquare_api import get_place_photos, search_attractions, search_destinations
from tools.mapbox_api import geocode_location
from tools.preference_scorer import score_preference_match
from tools.seed_data import all_destinations
from tools.weather import get_weather_forecast

_DEFAULT_SCORES = {
    "nature": 0.5,
    "adventure": 0.4,
    "food": 0.4,
    "nightlife": 0.2,
    "relaxation": 0.5,
    "culture": 0.4,
    "shopping": 0.3,
}


def _city_row(name: str, lat: float, lng: float, description: str, tags: list, scores: dict, source: str) -> dict:
    return {
        "name": name,
        "lat": lat,
        "lng": lng,
        "description": description,
        "tags": tags,
        "activity_scores": scores,
        "source": source,
    }


def _extra_named_cities(spec: TripSpec, existing: list[dict]) -> list[dict]:
    """Add towns the user named that the region search missed (e.g. Thalassery)."""
    have = {d["name"].lower() for d in existing}
    origin = (spec.origin_city or "").strip().lower()
    extra: list[dict] = []

    for dest in all_destinations():
        name = dest["name"]
        if name.lower() in have or name.lower() == origin:
            continue
        if name.lower() not in spec.raw_input.lower():
            continue
        extra.append(
            _city_row(
                name,
                dest["coordinates"]["lat"],
                dest["coordinates"]["lng"],
                dest.get("description") or name,
                dest.get("tags") or [],
                dest.get("activity_scores") or dict(_DEFAULT_SCORES),
                "seed",
            )
        )
        have.add(name.lower())

    from_to = re.search(
        r"\bfrom\s+[a-z][a-z .]+?\s+to\s+([a-z][a-z .]+?)(?:\s|,|$)",
        spec.raw_input.lower(),
    )
    named = [from_to.group(1).strip()] if from_to else []
    region_city = spec.destination_region.split(",")[0].strip()
    if region_city:
        named.append(region_city)
    for raw in named:
        key = raw.lower()
        if key in have or key == origin or key in {"india", "kerala"}:
            continue
        geo = geocode_location.invoke({"place_name": f"{raw}, India"})
        lat, lng = geo.get("lat"), geo.get("lng")
        if lat is None or lng is None:
            continue
        extra.append(
            _city_row(
                geo.get("name") or raw.title(),
                lat,
                lng,
                f"Trip base for {raw.title()}",
                [],
                dict(_DEFAULT_SCORES),
                geo.get("source", "mapbox"),
            )
        )
        have.add(key)
        have.add((geo.get("name") or raw).lower())
    return extra


def _match_named(candidates: list[Destination], wanted: list[str]) -> list[Destination]:
    by_lower = {c.name.lower(): c for c in candidates}
    picked: list[Destination] = []
    seen: set[str] = set()
    for name in wanted:
        match = by_lower.get(name.lower())
        if match and match.name not in seen:
            picked.append(match)
            seen.add(match.name)
    return picked


def destination_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    prefs = spec.preferences.as_dict()
    disruptions = state.get("disruptions", [])

    closed_or_risky_names = {
        d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)
    }

    raw_destinations = search_destinations.invoke({"region": spec.destination_region})
    raw_destinations = list(raw_destinations) + _extra_named_cities(spec, raw_destinations)

    candidates: list[Destination] = []
    for d in raw_destinations:
        score_result = score_preference_match.invoke(
            {"preferences": prefs, "category_scores": d["activity_scores"]}
        )
        weather = get_weather_forecast.invoke({"destination": d["name"], "day_offset": 0})
        is_disrupted = d["name"] in closed_or_risky_names

        preference_score = score_result["score"]
        if is_disrupted:
            preference_score *= 0.15

        candidates.append(
            Destination(
                name=d["name"],
                region=spec.destination_region,
                coordinates=Coordinates(lat=d["lat"], lng=d["lng"]),
                preference_score=preference_score,
                description=d["description"],
                tags=d["tags"],
                weather_summary=weather["summary"],
                weather_risk=weather["rain_risk"] or is_disrupted,
            )
        )

    candidates.sort(key=lambda d: d.preference_score, reverse=True)

    max_destinations = max(1, min(spec.constraints.max_destinations, max(1, spec.duration_days // 2)))
    viable = [c for c in candidates if c.name not in closed_or_risky_names]
    named = [c for c in viable if c.name.lower() in spec.raw_input.lower()]
    if named:
        max_destinations = max(max_destinations, min(len(named), spec.constraints.max_destinations))
    origin = (spec.origin_city or "").strip().lower()
    if origin:
        named = [c for c in named if c.name.lower() != origin]
        viable = [c for c in viable if c.name.lower() != origin]
    selected = (named or viable or candidates)[:max_destinations]
    llm_note = ""

    decision = llm_decide(
        get_llm(),
        tools=[search_destinations, search_attractions, get_weather_forecast, score_preference_match, get_place_photos],
        system=(
            "You are Odyssey's Destination Discovery agent — the explorer, not the scheduler. "
            "ROLE: pick which cities to keep as overnight/sightseeing stops. "
            "YOU DECIDE: a short list of city names from the heuristic ranking. "
            "YOU MUST NOT: invent a day plan, pick hotels, or choose travel order (Mobility does A*). "
            "Prefer cities the user named. Never pick a disrupted city. "
            "origin/home city is how they arrive — not a sightseeing stop. "
            "Call tools if you need weather, attractions, or preference scores. "
            "Then reply ONLY with JSON: {\"selected\": [\"City\", ...], \"reasoning\": \"...\"}."
        ),
        user=(
            f"Region: {spec.destination_region}. Duration: {spec.duration_days} days. "
            f"Max destinations: {max_destinations}. Preferences: {prefs}. "
            f"Disrupted (avoid): {sorted(closed_or_risky_names)}. "
            f"Heuristic ranking: {[{'name': d.name, 'score': d.preference_score, 'weather': d.weather_summary} for d in candidates[:8]]}."
        ),
    )
    if decision.get("selected"):
        wanted = [
            n
            for n in decision["selected"]
            if n not in closed_or_risky_names and str(n).lower() != origin
        ]
        picked = _match_named(candidates, [str(n) for n in wanted])
        if picked:
            selected = picked[:max_destinations]
            llm_note = f" LLM selected {', '.join(d.name for d in selected)}"
            if decision.get("_tool_calls"):
                llm_note += f" after tools {decision['_tool_calls']}"
            if decision.get("reasoning"):
                llm_note += f" ({decision['reasoning'][:180]})"

    activities_by_dest: dict[str, list[Activity]] = {}
    for dest in selected:
        acts_raw = search_attractions.invoke(
            {
                "destination": dest.name,
                "category": None,
                "lat": dest.coordinates.lat,
                "lng": dest.coordinates.lng,
            }
        )
        acts: list[Activity] = []
        for a in acts_raw:
            if a.get("name") in closed_or_risky_names or a.get("id") in closed_or_risky_names:
                continue
            if dest.name in closed_or_risky_names:
                continue
            category_score = score_preference_match.invoke(
                {"preferences": prefs, "category_scores": {a["category"]: 1.0}}
            )
            coords = a.get("coordinates")
            acts.append(
                Activity(
                    id=a["id"],
                    name=a["name"],
                    destination=dest.name,
                    category=a["category"],
                    duration_minutes=a["duration_minutes"],
                    cost_inr=a["cost_inr"],
                    rating=a["rating"],
                    opening_hour=a["opening_hour"],
                    closing_hour=a["closing_hour"],
                    preference_score=max(a.get("preference_score", 0.5), category_score["score"]),
                    coordinates=Coordinates(lat=coords["lat"], lng=coords["lng"]) if coords and coords.get("lat") is not None else dest.coordinates,
                    is_closed=False,
                )
            )
        activities_by_dest[dest.name] = acts

    names = ", ".join(d.name for d in selected)
    disrupted_note = f" (demoted {len(closed_or_risky_names)} disrupted option(s))" if closed_or_risky_names else ""
    source = f" via {llm_provider_name()} tool-calling" if llm_note else " via preference scorer"
    message = (
        f"Destination Agent: ranked {len(candidates)} candidates, selected [{names}] "
        f"by preference match{disrupted_note}{source}.{llm_note}"
    )

    return {
        "candidate_destinations": candidates,
        "selected_destinations": selected,
        "candidate_activities": activities_by_dest,
        "excluded_activity_ids": [],
        "agent_messages": [message],
    }
