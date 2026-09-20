"""Trip Analyst — turn a natural-language request into a TripSpec.

Role: first specialist in the graph. Parse what the traveller asked; do not
pick sights, hotels, or a day plan.

Decides: destination_region, origin_city (arrival only), duration, travellers,
budget, preference weights, pace, missing-field notes.

Computes: regex/keyword heuristic when no LLM is configured.

Uses: Mapbox `geocode_location`, `validate_trip_schema`. Never calls Foursquare,
hotels, or A*.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import ValidationError

from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import PreferenceWeights, TripConstraints, TripSpec
from orchestration.state import TripState
from tools.mapbox_api import geocode_location

SYSTEM_PROMPT = """You are Odyssey's Trip Analyst — the parser, not the planner.

ROLE: Convert the user's natural-language request into a TripSpec. Later agents
choose cities, routes, hotels, and the hour-by-hour schedule.

YOU DECIDE:
- destination_region: a geocode-able city or region ("Delhi, India", "Ooty, India",
  "Kerala, India"). Use the place they asked to visit, not a default.
- origin_city: home/arrival city when they say "from X to Y". That is how they
  arrive — not a sightseeing stop.
- duration_days, travellers, budget_inr, preference weights (0-1) for nature,
  adventure, food, nightlife, relaxation, culture, shopping.
- constraints.pace and max_destinations when they hint at them.

YOU MUST NOT: invent a city list, hotel names, flight numbers, or a day plan.
Do not default destination_region to Kerala unless they asked for Kerala or a
Kerala town. If they name a mode ("by train", "prefer flights"), leave it in
the request — Mobility will honour it.

MISSING FIELDS: assume a reasonable value and record it in needs_clarification."""


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


_REGION_KEYWORDS = [
    "kerala",
    "goa",
    "himachal pradesh",
    "rajasthan",
    "kashmir",
    "karnataka",
    "tamil nadu",
    "delhi",
    "mumbai",
    "chennai",
    "bangalore",
    "bengaluru",
]

_ORIGIN_CITIES = {
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "mumbai": "Mumbai",
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "chennai": "Chennai",
    "madras": "Chennai",
    "hyderabad": "Hyderabad",
    "kolkata": "Kolkata",
    "pune": "Pune",
    "goa": "Goa",
    "kochi": "Kochi",
    "cochin": "Kochi",
}


def _title_city(raw: str) -> str | None:
    key = raw.strip().lower()
    if not key or len(key) < 3:
        return None
    return _ORIGIN_CITIES.get(key) or raw.strip().title()


def _extract_origin_city(text: str) -> str | None:
    lowered = text.lower()
    from_to = re.search(r"\bfrom\s+([a-z][a-z .]+?)\s+to\s+([a-z][a-z .]+?)(?:\s|,|$)", lowered)
    if from_to:
        origin = _title_city(from_to.group(1))
        dest = _title_city(from_to.group(2))
        if origin and dest and origin.lower() != dest.lower():
            return origin
    named = re.search(r"\bfrom\s+([a-z][a-z .]+?)(?:\s|,|$)", lowered)
    if named:
        return _title_city(named.group(1))
    return None

_NOT_CITIES = {
    "somewhere",
    "there",
    "india",
    "this",
    "that",
    "nice",
    "the",
    "and",
    "for",
    "with",
    "from",
    "days",
    "day",
    "trip",
    "holiday",
    "vacation",
    "somewhere nice",
}


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
    from_to = re.search(r"\bfrom\s+([a-z][a-z .]+?)\s+to\s+([a-z][a-z .]+?)(?:\s|,|$)", text)
    dest_city = _title_city(from_to.group(2)) if from_to else None
    if dest_city:
        destination_region = f"{dest_city}, India"
    else:
        region_hit = next((c for c in _REGION_KEYWORDS if c in text), None)
        city_trip = re.search(r"(\d+)\s*-?\s*day(?:s)?\s+([a-z][a-z .]{2,}?)\s+trip\b", text)
        trip_city = _title_city(city_trip.group(2)) if city_trip else None
        if trip_city and trip_city.lower() in _NOT_CITIES:
            trip_city = None
        mentioned = re.search(
            r"\b(?:in|to|visit|visiting)\s+([a-z][a-z .]{2,}?)(?:\s+for|\s+with|\s+and|\s*,|$)",
            text,
        )
        mentioned_city = _title_city(mentioned.group(1)) if mentioned else None
        if mentioned_city and mentioned_city.lower() in _NOT_CITIES:
            mentioned_city = None
        if region_hit:
            destination_region = region_hit.title() + ", India"
        elif trip_city:
            destination_region = f"{trip_city}, India"
        elif mentioned_city:
            destination_region = f"{mentioned_city}, India"
        else:
            needs_clarification.append("region_assumed_default_kerala")

    origin_city = _extract_origin_city(text)
    if origin_city and origin_city.lower() in destination_region.lower():
        origin_city = None

    return TripSpec(
        destination_region=destination_region,
        duration_days=duration,
        travellers=travellers,
        budget_inr=budget,
        preferences=prefs,
        constraints=TripConstraints(),
        origin_city=origin_city,
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
                "destination_region, duration_days, travellers, budget_inr, origin_city "
                "(home city or null), "
                "preferences (nature,adventure,food,nightlife,relaxation,culture,shopping), "
                "constraints (max_daily_travel_hours, max_destinations, pace), "
                "needs_clarification (list of strings)."
            ),
            user=f"User request: {raw_input}",
        )
        if decision.get("destination_region"):
            spec = TripSpec.model_validate({**decision, "raw_input": raw_input})
            if not spec.origin_city:
                spec.origin_city = _extract_origin_city(raw_input)
            return spec
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
    origin_bit = f", arriving from {spec.origin_city}" if spec.origin_city else ""
    source = f" via {llm_provider_name()} tool-calling" if used_llm else " via offline heuristics"
    message = (
        f"Trip Analyst: parsed request -> {spec.destination_region}{origin_bit}, {spec.duration_days} days, "
        f"₹{spec.budget_inr:.0f} for {spec.travellers} traveller(s){clarif}{source}."
    )

    return {"trip_spec": spec, "agent_messages": [message]}
