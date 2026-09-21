"""Trip Analyst — turn a natural-language request into a TripSpec.

Role: first specialist in the graph. Parse what the traveller asked; do not
pick sights, hotels, or a day plan.

Decides (LLM): destination_region, origin_city (arrival only), duration, travellers,
budget, start date, preference weights, pace, and which of duration / travellers /
budget the traveller did not state (so they can be told what was assumed).
If no destination was given it asks a question instead of guessing.

Computes: validates the reply against TripSpec and keeps odd values in range.
There is no fallback parser: without a language model it stops with a clear error,
because a guessed destination is worse than none.

Uses: Mapbox `geocode_location`, `validate_trip_schema`. Never calls Foursquare,
hotels, or A*.
"""
from __future__ import annotations

from datetime import date, timedelta

from langchain_core.tools import tool
from pydantic import ValidationError

from llm import MODEL_UNAVAILABLE, agent_trace, get_llm, llm_decide
from models.schemas import TripSpec
from orchestration.state import TripState
from tools.mapbox_api import geocode_location

SYSTEM_PROMPT = """You are Odyssey's Trip Analyst — the parser, not the planner.

ROLE: Convert the user's natural-language request into a TripSpec. Later agents
choose cities, routes, hotels, and the hour-by-hour schedule.

Reply ONLY with JSON:
{"destination_region": "a geocode-able city or region, e.g. \\"Delhi, India\\", \\"Ooty, India\\", \\"Kerala, India\\"; \\"\\" if they did not say where",
 "clarifying_question": "ONE short question. Ask it when destination_region would be empty, and also when the place is fictional or not a real destination (a geocoder can still match a name to some real spot), or when an ordinary holiday there is impossible (Antarctica, the Moon). Then leave destination_region empty and say plainly why, offering a real alternative",
 "origin_city": "home/arrival city if they say they travel from somewhere, else null (it is how they arrive, not a stop)",
 "duration_days": int, "travellers": int, "budget_inr": number,
 "start_date": "YYYY-MM-DD if they gave a date, else null",
 "preferences": {"nature": 0-1, "adventure": 0-1, "food": 0-1, "nightlife": 0-1, "relaxation": 0-1, "culture": 0-1, "shopping": 0-1},
 "constraints": {"pace": "relaxed|moderate|packed", "max_daily_travel_hours": number, "max_destinations": int},
 "assumed": ["duration", "travellers", "budget"]  (only the ones the user did NOT state and you filled in)}

Use the place they asked for; never default it. "A couple" is 2 travellers, "family of 4" is 4, "with 3 friends" is 4.
Give preferences weight 0.5 when they say nothing about a theme and 0.9 for what they stress.
Do not invent a city list, hotels, flight numbers, or a day plan. If they name a transport mode, leave it
in the request — Mobility will honour it."""

_ASSUMABLE = {"duration", "travellers", "budget"}


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


def _finalize(spec: TripSpec) -> TripSpec:
    """Every plan needs a calendar, and a model's odd values (a 24h "daily" travel cap) need bounding."""
    try:
        valid = spec.start_date and date.fromisoformat(spec.start_date) >= date.today()
    except ValueError:
        valid = False
    if not valid:
        spec.start_date = (date.today() + timedelta(days=14)).isoformat()
        if "start_date" not in spec.needs_clarification:
            spec.needs_clarification.append("start_date")
    c = spec.constraints
    spec.constraints = c.model_copy(
        update={
            "max_daily_travel_hours": min(8.0, max(2.0, float(c.max_daily_travel_hours))),
            "max_destinations": min(5, max(1, int(c.max_destinations))),
            "pace": c.pace if c.pace in {"relaxed", "moderate", "packed"} else "moderate",
        }
    )
    return spec


def _without_nulls(values) -> dict:
    """A model often writes null for a value it has no opinion on; leave those to the defaults."""
    return {k: v for k, v in values.items() if v is not None} if isinstance(values, dict) else {}


def _llm_extract(raw_input: str) -> tuple[TripSpec | None, dict]:
    """The spec, and the raw decision so the caller can report which tools the model called."""
    llm = get_llm()
    if llm is None:
        return None, {}
    decision = llm_decide(
        llm,
        tools=[geocode_location, validate_trip_schema],
        system=SYSTEM_PROMPT,
        user=f"Today's date is {date.today().isoformat()}. User request: {raw_input}",
    )
    if not decision or not (decision.get("destination_region") or decision.get("clarifying_question")):
        return None, decision
    try:
        duration = max(1, min(30, int(decision.get("duration_days") or 5)))
        spec = TripSpec.model_validate(
            {
                "destination_region": str(decision.get("destination_region") or "").strip(),
                "clarifying_question": decision.get("clarifying_question") or None,
                "origin_city": decision.get("origin_city") or None,
                "duration_days": duration,
                "travellers": max(1, int(decision.get("travellers") or 1)),
                "budget_inr": float(decision.get("budget_inr") or 40000),
                "start_date": decision.get("start_date") or None,
                "preferences": _without_nulls(decision.get("preferences")),
                "constraints": _without_nulls(decision.get("constraints")),
                "needs_clarification": [a for a in decision.get("assumed") or [] if a in _ASSUMABLE],
                "raw_input": raw_input,
            }
        )
    except (ValidationError, TypeError, ValueError):
        return None, decision
    return _finalize(spec), decision


def trip_analyst_node(state: TripState) -> dict:
    raw_input = state["raw_input"]
    spec, decision = _llm_extract(raw_input)
    if spec is None:
        raise RuntimeError(MODEL_UNAVAILABLE)
    meta = agent_trace(decision)

    if not spec.destination_region:
        question = spec.clarifying_question or "Where would you like to go?"
        return {
            "trip_spec": spec,
            "assistant_reply": question,
            "agent_meta": meta,
            "agent_messages": [f"Trip Analyst: no destination given — asking: {question}"],
        }

    origin_bit = f", arriving from {spec.origin_city}" if spec.origin_city else ""
    assumed = f" (assumed: {', '.join(spec.needs_clarification)})" if spec.needs_clarification else ""
    message = (
        f"Trip Analyst: parsed request -> {spec.destination_region}{origin_bit}, {spec.duration_days} days, "
        f"₹{spec.budget_inr:.0f} for {spec.travellers} traveller(s){assumed}."
    )
    return {"trip_spec": spec, "assistant_reply": None, "agent_meta": meta, "agent_messages": [message]}
