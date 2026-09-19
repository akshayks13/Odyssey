"""
Agent 1 — Trip Analyst

Converts a free-form natural-language travel request into a structured
TripSpec. Uses the LLM's structured-output mode when an LLM is configured;
otherwise falls back to a deterministic regex/keyword extractor so the
pipeline always produces a usable TripSpec.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import ValidationError

from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import PreferenceWeights, TripConstraints, TripSpec
from orchestration.state import TripState
from tools.mapbox_api import geocode_location

SYSTEM_PROMPT = """You are the Trip Analyst agent in Odyssey, a multi-agent travel
planning system. Extract a structured trip specification from the user's
natural-language travel request. Infer preference weights (0-1) for nature,
adventure, food, nightlife, relaxation, culture, and shopping based on the
tone and explicit mentions in the request. If budget, duration, or number of
travellers is not stated, make a reasonable assumption and add a short note
about it to `needs_clarification`. Keep `destination_region` as a broad,
geocode-able region name, e.g. "Kerala, India"."""


@tool(parse_docstring=True)
def validate_trip_schema(spec: dict) -> dict:
    """Validate a candidate trip specification against the TripSpec schema.

    Args:
        spec: Dict of TripSpec fields (destination_region, duration_days,
            travellers, budget_inr, preferences, constraints, ...).

    Returns:
        dict with valid (bool) and errors (list of strings).
    """
    try:
        TripSpec.model_validate(spec)
        return {"valid": True, "errors": []}
    except ValidationError as exc:
        return {"valid": False, "errors": [e["msg"] for e in exc.errors()]}


_REGION_KEYWORDS = ["kerala", "goa", "himachal pradesh", "rajasthan", "kashmir", "karnataka", "tamil nadu"]

_PREFERENCE_KEYWORDS = {
    "nature": ["nature", "scenic", "hills", "greenery", "mountains", "forest"],
    "adventure": ["adventure", "trek", "hiking", "paraglid", "rafting", "thrill"],
    "food": ["food", "cuisine", "culinary", "eat"],
    "nightlife": ["nightlife", "party", "club", "bar"],
    "relaxation": ["relax", "chill", "calm", "peaceful", "slow", "leisure"],
    "culture": ["culture", "heritage", "history", "temple", "tradition"],
    "shopping": ["shopping", "market", "mall"],
}


def _heuristic_extract(raw_input: str) -> TripSpec:
    """Deterministic fallback extractor used when no LLM is configured."""
    text = raw_input.lower()
    needs_clarification: list[str] = []

    duration = 5
    m = re.search(r"(\d+)\s*-?\s*day", text)
    if m:
        duration = int(m.group(1))
    else:
        needs_clarification.append("duration_assumed_default_5_days")

    travellers = 1
    m = re.search(r"(\d+)\s*(people|friends|travellers|travelers|persons|pax|adults)", text)
    if m:
        count, word = int(m.group(1)), m.group(2)
        # "with N friends" implies the speaker + N friends
        travellers = count + 1 if word == "friends" else count
    elif "family" in text:
        travellers = 4
    elif "solo" in text:
        travellers = 1
    else:
        needs_clarification.append("travellers_assumed_default_1")

    budget = 40000.0
    m = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*)", text)
    if not m:
        m = re.search(r"(\d[\d,]*)\s*(?:₹|rs\.?|inr|rupees?)", text)
    if not m:
        m = re.search(r"budget\D{0,10}?(\d[\d,]*)", text)
    if m:
        budget = float(m.group(1).replace(",", ""))
    else:
        needs_clarification.append("budget_assumed_default_40000_inr")

    prefs = PreferenceWeights()
    updates = {}
    for key, words in _PREFERENCE_KEYWORDS.items():
        if any(w in text for w in words):
            updates[key] = 0.9
    if updates:
        prefs = prefs.model_copy(update=updates)

    destination_region = "Kerala, India"
    for candidate in _REGION_KEYWORDS:
        if candidate in text:
            destination_region = candidate.title() + ", India"
            break
    else:
        needs_clarification.append("region_assumed_default_kerala")

    return TripSpec(
        destination_region=destination_region,
        duration_days=duration,
        travellers=travellers,
        budget_inr=budget,
        preferences=prefs,
        constraints=TripConstraints(),
        needs_clarification=needs_clarification,
        raw_input=raw_input,
    )


def _llm_extract(raw_input: str) -> TripSpec | None:
    llm = get_llm()
    if llm is None:
        return None
    try:
        decision = llm_decide(
            llm,
            tools=[geocode_location, validate_trip_schema],
            system=(
                SYSTEM_PROMPT
                + " You MUST call geocode_location on the destination region and "
                "validate_trip_schema on your extracted spec. "
                "After tools return, reply with JSON matching TripSpec fields: "
                "destination_region, duration_days, travellers, budget_inr, "
                "preferences (nature,adventure,food,nightlife,relaxation,culture,shopping), "
                "constraints (max_daily_travel_hours, max_destinations, pace), "
                "needs_clarification (list of strings)."
            ),
            user=f"User request: {raw_input}",
        )
        if decision.get("destination_region"):
            return TripSpec.model_validate({**decision, "raw_input": raw_input})
        structured = llm.with_structured_output(TripSpec)
        spec = structured.invoke(f"{SYSTEM_PROMPT}\n\nUser request: {raw_input}")
        spec.raw_input = raw_input
        return spec
    except Exception:
        return None


def trip_analyst_node(state: TripState) -> dict:
    raw_input = state["raw_input"]

    used_llm = False
    spec = _llm_extract(raw_input)
    if spec is not None:
        used_llm = True
    else:
        spec = _heuristic_extract(raw_input)

    geocode_location.invoke({"place_name": spec.destination_region})

    clarif = f" (assumed: {', '.join(spec.needs_clarification)})" if spec.needs_clarification else ""
    source = f" via {llm_provider_name()} tool-calling" if used_llm else " via offline heuristics"
    message = (
        f"Trip Analyst: parsed request -> {spec.destination_region}, {spec.duration_days} days, "
        f"₹{spec.budget_inr:.0f} for {spec.travellers} traveller(s){clarif}{source}."
    )

    return {"trip_spec": spec, "agent_messages": [message]}
