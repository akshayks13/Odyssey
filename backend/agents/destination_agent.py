"""Destination Discovery — pick which cities to keep and load their sights.

Role: explorer. Rank candidate cities for the parsed TripSpec; do not order
them on a map and do not book hotels.

Decides: which 1..N cities to keep (never a closed/weather-risky city; never
the origin/home city as a sightseeing stop). Checks drive times with
`get_directions` so the stops are not a day apart.

Computes: weighted cosine preference score per city; weather; attraction
list per selected city. Honours cities the user pinned or excluded.

Tools the model can call: `search_attractions`, `get_directions`. The code itself uses
`search_destinations` (LLM places for any region), weather for the trip dates,
and `score_preference_match`, because it always needs those.
"""
from __future__ import annotations

from algorithms.planning import activity_is_excluded
from llm import MODEL_UNAVAILABLE, agent_trace, get_llm, llm_decide
from models.schemas import Activity, Coordinates, Destination, DisruptionType, EditLocks, TripSpec
from orchestration.state import TripState
from tools import pmap
from tools.foursquare_api import search_attractions, search_destinations
from tools.mapbox_api import geocode_location, get_directions, haversine_km
from tools.preference_scorer import score_preference_match
from tools.weather import trip_weather

_DEFAULT_SCORES = {
    "nature": 0.5,
    "adventure": 0.4,
    "food": 0.4,
    "nightlife": 0.2,
    "relaxation": 0.5,
    "culture": 0.4,
    "shopping": 0.3,
}


def _country_of(region: str) -> str:
    return region.split(",")[-1].strip() if "," in region else "India"


def _within_a_days_drive(cities: list[Destination], max_daily_hours: float) -> list[Destination]:
    """Keep the best-ranked city and those a day's drive from it (about 50 km/h), so stops are not hours apart."""
    if not cities:
        return cities
    first = cities[0].coordinates
    return [c for c in cities if haversine_km(first.lat, first.lng, c.coordinates.lat, c.coordinates.lng) <= max_daily_hours * 50]


def _match_named(candidates: list[Destination], wanted: list[str]) -> list[Destination]:
    by_lower = {c.name.lower(): c for c in candidates}
    picked: list[Destination] = []
    for name in wanted:
        match = by_lower.get(name.lower())
        if match and match not in picked:
            picked.append(match)
    return picked


def _resolve_pinned(names: list[str], candidates: list[Destination], spec: TripSpec) -> list[Destination]:
    """Cities the user asked for: the discovered ones, or geocoded if discovery didn't list them."""
    by_lower = {c.name.lower(): c for c in candidates}
    out: list[Destination] = []
    for name in names:
        hit = by_lower.get(name.lower())
        if hit is None:
            geo = geocode_location.invoke({"place_name": f"{name}, {_country_of(spec.destination_region)}"})
            if geo["lat"] is None:
                continue
            hit = Destination(
                name=geo.get("name") or name.title(),
                region=spec.destination_region,
                coordinates=Coordinates(lat=geo["lat"], lng=geo["lng"]),
                description=f"Added at your request ({name.title()})",
            )
        if hit not in out:
            out.append(hit)
    return out


def destination_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    prefs = spec.preferences.as_dict()
    disruptions = state.get("disruptions", [])
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    sent_back = [d.reason for d in state.get("replan_directives", []) if d.target_agent == "destination_agent"]

    closed_or_risky_names = {
        d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)
    }
    origin = (spec.origin_city or "").strip().lower()
    avoid = [c for d in state.get("replan_directives", []) if d.target_agent == "destination_agent" for c in d.constraints.get("avoid", [])]
    excluded_names = {n.lower() for n in [*locks.excluded_cities, *closed_or_risky_names, *avoid]} | ({origin} if origin else set())

    raw_destinations = search_destinations.invoke({"region": spec.destination_region})
    if not raw_destinations:  # the model listed no places: stop, don't invent a one-city trip
        raise RuntimeError(MODEL_UNAVAILABLE)

    def build_candidate(d: dict) -> Destination:
        score = score_preference_match.invoke({"preferences": prefs, "category_scores": d["activity_scores"]})["score"]
        weather = trip_weather(d["name"], spec.start_date or "", spec.duration_days)
        rainy_days = len(weather["rainy_dates"])
        disrupted = d["name"] in closed_or_risky_names
        return Destination(
            name=d["name"],
            region=spec.destination_region,
            coordinates=Coordinates(lat=d["lat"], lng=d["lng"]),
            preference_score=score * (0.15 if disrupted else 1.0),
            description=d["description"],
            tags=d["tags"],
            weather_summary=weather["summary"],
            weather_risk=rainy_days >= max(1, round(0.4 * spec.duration_days)) or disrupted,
        )

    candidates = [c for c in pmap(build_candidate, raw_destinations, workers=8) if c is not None]
    candidates.sort(key=lambda d: d.preference_score, reverse=True)

    max_destinations = max(1, min(spec.constraints.max_destinations, max(1, spec.duration_days // 2)))
    viable = [c for c in candidates if c.name.lower() not in excluded_names]
    named = [c for c in viable if c.name.lower() in spec.raw_input.lower()]
    # The place they asked for is always a stop. Discovery may call it "New Delhi", so match by containment
    # and take the best-ranked one.
    requested = spec.destination_region.split(",")[0].strip().lower()
    anchor = next((c for c in viable if requested and requested in c.name.lower()), None)
    if anchor and anchor not in named:
        named.insert(0, anchor)
    if named:
        max_destinations = max(max_destinations, min(len(named), spec.constraints.max_destinations))
    # Never keep more cities than there are days: the schedule can only cover one city per day, and a
    # city the schedule drops would still be routed to, priced, and drawn on the map.
    max_destinations = min(max_destinations, spec.duration_days)
    selected = (named or _within_a_days_drive(viable, spec.constraints.max_daily_travel_hours) or candidates)[:max_destinations]
    llm_note = ""
    decision: dict = {}

    if locks.pinned_cities:
        # The user chose the cities. Keep them; only replace one that has become unavailable.
        keep = [c for c in _resolve_pinned(locks.pinned_cities, candidates, spec) if c.name.lower() not in excluded_names]
        selected = (keep + [c for c in viable if c not in keep])[: max(len(locks.pinned_cities), 1)] or selected
        llm_note = " Kept your chosen cities."
    else:
        decision = llm_decide(
            get_llm(),
            tools=[search_attractions, get_directions],
            system=(
                "You are Odyssey's Destination Discovery agent — the explorer, not the scheduler. "
                "ROLE: pick which cities to keep as overnight/sightseeing stops. "
                "YOU DECIDE: a short list of city names from the heuristic ranking. "
                "YOU MUST NOT: invent a day plan, pick hotels, or choose travel order (Mobility does A*). "
                "Prefer cities the user named. Never pick a disrupted city. "
                "Never pick two names for the same city or neighbourhood (Fort Kochi and Kochi are one place). "
                "origin/home city is how they arrive — not a sightseeing stop. "
                "Use get_directions to check drive times: stops should be close enough that travel does not "
                "eat the trip, and each stop needs enough sights for the days it will get. "
                "Each candidate has the weather for the trip dates: avoid a stop that is mostly rain "
                "if a comparable one is dry. "
                "Call search_attractions on a shortlisted city to see whether it has enough to fill its days. "
                "Then reply ONLY with JSON: {\"selected\": [\"City\", ...], \"reasoning\": \"...\"}."
            ),
            user=(
                f"Region: {spec.destination_region}. Duration: {spec.duration_days} days. "
                f"Max destinations: {max_destinations}. Max travel per day: {spec.constraints.max_daily_travel_hours}h. "
                f"Preferences: {prefs}. Disrupted (avoid): {sorted(closed_or_risky_names)}. "
                + (f"The reviewer sent this back because: {'; '.join(sent_back)}. " if sent_back else "")
                + f"Heuristic ranking: {[{'name': d.name, 'score': d.preference_score, 'weather': d.weather_summary} for d in candidates[:8]]}."
            ),
        )
        if decision.get("selected"):
            wanted = [str(n) for n in decision["selected"] if str(n).lower() not in excluded_names]
            picked = _match_named(viable, wanted)
            picked = (named + [p for p in picked if p not in named])[:max(max_destinations, len(named))]  # never drop what they named
            if picked:
                selected = picked
                llm_note = f" LLM selected {', '.join(d.name for d in selected)}"
                if decision.get("reasoning"):
                    llm_note += f" ({decision['reasoning'][:180]})"

    raw_acts = pmap(
        lambda dest: search_attractions.invoke(
            {"destination": dest.name, "category": None, "lat": dest.coordinates.lat, "lng": dest.coordinates.lng}
        ),
        selected,
        workers=4,
    )

    activities_by_dest: dict[str, list[Activity]] = {}
    for dest, acts_raw in zip(selected, raw_acts):
        acts: list[Activity] = []
        for a in acts_raw or []:
            if a.get("name") in closed_or_risky_names or a.get("id") in closed_or_risky_names:
                continue
            if dest.name in closed_or_risky_names or activity_is_excluded(a["name"], locks):
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
                    source=str(a.get("source") or ""),
                )
            )
        activities_by_dest[dest.name] = acts

    seen: set[str] = set()  # a sight listed under two stops is visited once, at the first
    for name, acts in activities_by_dest.items():
        activities_by_dest[name] = [a for a in acts if a.name.strip().lower() not in seen and not seen.add(a.name.strip().lower())]

    names = ", ".join(d.name for d in selected)
    disrupted_note = f" (demoted {len(closed_or_risky_names)} disrupted option(s))" if closed_or_risky_names else ""
    message = (
        f"Destination Agent: ranked {len(candidates)} candidates, selected [{names}] "
        f"by preference match{disrupted_note}.{llm_note}"
    )

    return {
        "candidate_destinations": candidates,
        "selected_destinations": selected,
        "candidate_activities": activities_by_dest,
        "excluded_activity_ids": [],
        "agent_meta": agent_trace(decision, algorithms=["weighted cosine preference scoring"]),
        "agent_messages": [message],
    }
