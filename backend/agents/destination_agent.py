"""
Agent 2 — Destination Discovery

Finds candidate destinations and activities within the trip's region,
ranks them by preference match (deterministic tool: cosine similarity),
and filters closed/disrupted activities when this agent is re-invoked
mid-replan (e.g. after a closure or weather disruption).
"""
from __future__ import annotations

from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import Activity, Coordinates, Destination, DisruptionType
from orchestration.state import TripState
from tools.foursquare_api import get_place_photos, search_attractions, search_destinations
from tools.preference_scorer import score_preference_match
from tools.weather import get_weather_forecast


def destination_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    prefs = spec.preferences.as_dict()
    disruptions = state.get("disruptions", [])

    closed_or_risky_names = {
        d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)
    }

    raw_destinations = search_destinations.invoke({"region": spec.destination_region})

    candidates: list[Destination] = []
    for d in raw_destinations:
        score_result = score_preference_match.invoke(
            {"preferences": prefs, "category_scores": d["activity_scores"]}
        )
        weather = get_weather_forecast.invoke({"destination": d["name"], "day_offset": 0})
        is_disrupted = d["name"] in closed_or_risky_names

        preference_score = score_result["score"]
        if is_disrupted:
            # Heavily penalize (but don't fully exclude) disrupted destinations
            # so the ranking naturally demotes them below viable alternatives.
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
    selected = (viable or candidates)[:max_destinations]
    llm_note = ""

    # LLM ReAct: the model must call the search/weather/scorer tools itself
    # and then pick which cities to keep. Tools remain the source of scores;
    # the LLM supplies the selection reasoning (and can demote weather-risky
    # or disrupted places more aggressively than the heuristic cutoff).
    decision = llm_decide(
        get_llm(),
        tools=[search_destinations, search_attractions, get_weather_forecast, score_preference_match, get_place_photos],
        system=(
            "You are Odyssey's Destination Discovery agent. Call tools to inspect "
            "destinations, weather, attractions and preference scores. Then reply "
            "ONLY with JSON: {\"selected\": [\"City\", ...], \"reasoning\": \"...\"}. "
            "Pick at most the requested max destinations. Never pick a disrupted city."
        ),
        user=(
            f"Region: {spec.destination_region}. Duration: {spec.duration_days} days. "
            f"Max destinations: {max_destinations}. Preferences: {prefs}. "
            f"Disrupted (avoid): {sorted(closed_or_risky_names)}. "
            f"Heuristic ranking: {[{'name': d.name, 'score': d.preference_score, 'weather': d.weather_summary} for d in candidates[:8]]}."
        ),
    )
    if decision.get("selected"):
        wanted = [n for n in decision["selected"] if n not in closed_or_risky_names]
        picked = [c for c in candidates if c.name in wanted]
        picked.sort(key=lambda d: wanted.index(d.name) if d.name in wanted else 99)
        if picked:
            selected = picked[:max_destinations]
            llm_note = f" LLM selected {', '.join(d.name for d in selected)}"
            if decision.get("_tool_calls"):
                llm_note += f" after tools {decision['_tool_calls']}"
            if decision.get("reasoning"):
                llm_note += f" ({decision['reasoning'][:180]})"

    activities_by_dest: dict[str, list[Activity]] = {}
    for dest in selected:
        acts_raw = search_attractions.invoke({"destination": dest.name, "category": None})
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
