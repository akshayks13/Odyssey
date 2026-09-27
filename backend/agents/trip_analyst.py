"""Trip Analyst — turn the traveller's sentence into a TripSpec.

Role: parser, first in the pipeline. It does not pick cities, hotels or a schedule.

How: keyword and pattern rules (no language model). It reads
    region      a region in the dataset, or one of its cities ("Munnar" -> Kerala)
    days        "5 days", "5-day", "4 nights", "a week", "weekend"
    travellers  "for 4 people", "with 3 friends" (= 4), "family of 5", "couple", "solo"
    budget      "₹40,000", "Rs 40000", "40k", "1.5 lakh", "₹10,000 per person"
    interests   words per theme: "hills", "trek", "beach", "temple", "food"...
    pace        "relaxed", "slow", "packed", "busy"
    transport   "by train", "by road"
    start date  "2026-12-20", "20 December", "December 20", "in December"
Anything not stated is filled with a default and reported as an assumption. A request for a place the
dataset does not cover gets a question back instead of a guess.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from agents.common import trace
from models.schemas import EditLocks, PreferenceWeights, TripConstraints, TripSpec
from orchestration.state import TripState
from tools import world

DEFAULT_DAYS = 5
DEFAULT_TRAVELLERS = 2
DEFAULT_BUDGET_PER_PERSON = 20000.0

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1,
}
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

THEME_WORDS = {
    "nature": ["nature", "hill", "hills", "mountain", "tea", "waterfall", "wildlife", "forest", "greenery", "backwater", "backwaters", "scenic", "views", "lake"],
    "adventure": ["adventure", "trek", "trekking", "hike", "hiking", "rafting", "paragliding", "surf", "surfing", "safari", "zipline", "thrill", "kayak", "kayaking"],
    "food": ["food", "foodie", "cuisine", "seafood", "eat", "eating", "cooking", "spice", "spices"],
    "nightlife": ["nightlife", "party", "parties", "bar", "bars", "club", "clubs"],
    "relaxation": ["relax", "relaxing", "beach", "beaches", "spa", "ayurveda", "yoga", "calm", "peaceful", "houseboat"],
    "culture": ["culture", "cultural", "temple", "temples", "history", "historic", "heritage", "museum", "museums", "art", "kathakali", "architecture"],
    "shopping": ["shopping", "shop", "market", "markets", "souvenir", "souvenirs"],
}
PACE_WORDS = {
    "relaxed": ["relaxed", "slow", "leisurely", "easy", "laid-back", "laid back", "unhurried", "chill"],
    "packed": ["packed", "busy", "action-packed", "fast", "see everything", "as much as possible", "hectic"],
}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _WORD_NUMBERS[token]


def parse_days(text: str) -> int | None:
    m = re.search(rf"\b{_NUM}\s+weeks?\b", text)
    if m:
        return 7 * _to_int(m.group(1))
    if re.search(r"\bweek\b", text) and not re.search(rf"\b{_NUM}\s*[- ]?\s*(days?|nights?)\b", text):
        return 7
    if re.search(r"\bweekend\b", text):
        return 2
    m = re.search(rf"\b{_NUM}\s*[- ]?\s*(days?|nights?)\b", text)
    if m:
        n = _to_int(m.group(1))
        return n + 1 if m.group(2).startswith("night") else n
    return None


def parse_travellers(text: str) -> int | None:
    m = re.search(rf"\bwith\s+(?:my\s+)?{_NUM}\s+(friends?|colleagues|others|people|kids|children)\b", text)
    if m:
        return _to_int(m.group(1)) + 1
    m = re.search(rf"\bfamily of\s+{_NUM}\b", text) or re.search(rf"\bgroup of\s+{_NUM}\b", text)
    if m:
        return _to_int(m.group(1))
    m = re.search(rf"\b{_NUM}\s+(people|persons|adults|travellers|travelers|pax|of us|friends|members)\b", text)
    if m:
        return _to_int(m.group(1))
    m = re.search(rf"\bfor\s+{_NUM}\b(?!\s*(days?|nights?|k\b|lakh))", text)
    if m:
        return _to_int(m.group(1))
    if re.search(r"\b(couple|honeymoon|my (wife|husband|partner|girlfriend|boyfriend))\b", text):
        return 2
    if re.search(r"\b(solo|alone|by myself|just me)\b", text):
        return 1
    return None


def parse_budget(text: str, travellers: int) -> float | None:
    patterns = [
        r"(?:₹|rs\.?|inr|rupees)\s*([\d,]+(?:\.\d+)?)\s*(k|lakhs?|l)?\b",
        r"([\d,]+(?:\.\d+)?)\s*(k|lakhs?)\b",
        r"budget\s*(?:of|is|:)?\s*(?:around|about)?\s*([\d,]+(?:\.\d+)?)\s*(k|lakhs?)?",
        r"([\d,]+(?:\.\d+)?)\s*(?:rupees|inr)\b()",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit == "k":
            value *= 1_000
        elif unit.startswith("l"):
            value *= 100_000
        if value < 500:  # "for 4" or a stray number, not a budget
            continue
        tail = text[m.end(): m.end() + 25]
        if re.search(r"\b(per person|each|per head|pp|a head)\b", tail):
            value *= travellers
        return value
    return None


def parse_preferences(text: str) -> tuple[PreferenceWeights, list[str]]:
    words = set(re.findall(r"[a-z]+", text))
    stressed = [theme for theme, keys in THEME_WORDS.items() if words & set(keys)]
    if not stressed:
        return PreferenceWeights(), []
    return PreferenceWeights(**{theme: (0.9 if theme in stressed else 0.3) for theme in THEME_WORDS}), stressed


def parse_pace(text: str) -> str:
    for pace, words in PACE_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            return pace
    return "moderate"


def parse_mode(text: str) -> str | None:
    if re.search(r"\b(by|on the|take the|via)\s+(train|rail)\b|\btrains?\b", text):
        return "rail"
    if re.search(r"\b(by (road|car|taxi)|road trip|drive|driving)\b", text):
        return "road"
    return None


def parse_start_date(text: str, today: date) -> str | None:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        try:
            return date.fromisoformat(m.group(1)).isoformat()
        except ValueError:
            return None
    month_re = "|".join(_MONTHS + [m[:3] for m in _MONTHS])
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_re})\b", text) or re.search(rf"\b({month_re})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text)
    day, month = None, None
    if m:
        a, b = m.group(1), m.group(2)
        day, month_name = (int(a), b) if a.isdigit() else (int(b), a)
        month = next(i + 1 for i, name in enumerate(_MONTHS) if name.startswith(month_name[:3]))
    else:
        m = re.search(rf"\b(?:in|during|this|next)\s+({month_re})\b", text)
        if m:
            day, month = 10, next(i + 1 for i, name in enumerate(_MONTHS) if name.startswith(m.group(1)[:3]))
    if month is None:
        return None
    try:
        when = date(today.year, month, day)
        if when < today:
            when = date(today.year + 1, month, day)
    except ValueError:
        return None
    return when.isoformat()


def parse_request(raw_input: str, today: date | None = None) -> tuple[TripSpec, str | None]:
    """The TripSpec, and the transport mode the traveller asked for (if any)."""
    today = today or date.today()
    text = raw_input.lower().replace("’", "'")
    assumed: list[str] = []

    region = world.match_region(text) or ""
    days = parse_days(text)
    if days is None:
        days = DEFAULT_DAYS
        assumed.append("duration")
    travellers = parse_travellers(text)
    if travellers is None:
        travellers = DEFAULT_TRAVELLERS
        assumed.append("travellers")
    budget = parse_budget(text, travellers)
    if budget is None:
        budget = DEFAULT_BUDGET_PER_PERSON * travellers
        assumed.append("budget")
    start = parse_start_date(text, today)
    if start is None:
        start = (today + timedelta(days=14)).isoformat()
        assumed.append("start_date")
    preferences, _ = parse_preferences(text)
    pace = parse_pace(text)

    spec = TripSpec(
        destination_region=region,
        duration_days=max(1, min(30, days)),
        travellers=max(1, travellers),
        budget_inr=budget,
        preferences=preferences,
        constraints=TripConstraints(pace=pace, max_daily_travel_hours=5.0 if pace == "packed" else 4.0, max_destinations=4),
        start_date=start,
        needs_clarification=assumed,
        clarifying_question=None if region else (
            "This planner works offline and covers: " + ", ".join(r.split(",")[0] for r in world.regions())
            + ". Which of these would you like to visit, or which city there?"
        ),
        raw_input=raw_input,
    )
    return spec, parse_mode(text)


def trip_analyst_node(state: TripState) -> dict:
    spec, mode = parse_request(state["raw_input"])
    meta = trace(tools=["match_region"], algorithms=["keyword / pattern parsing"])

    if not spec.destination_region:
        return {
            "trip_spec": spec,
            "assistant_reply": spec.clarifying_question,
            "agent_meta": meta,
            "agent_messages": [f"Trip Analyst: no known destination in the request — asking: {spec.clarifying_question}"],
        }

    locks: EditLocks = (state.get("edit_locks") or EditLocks()).model_copy(deep=True)
    if mode:
        locks.preferred_mode = mode
    stressed = [k for k, v in spec.preferences.as_dict().items() if v >= 0.9]
    assumed = f" (assumed: {', '.join(spec.needs_clarification)})" if spec.needs_clarification else ""
    message = (
        f"Trip Analyst: {spec.destination_region}, {spec.duration_days} days from {spec.start_date}, "
        f"₹{spec.budget_inr:,.0f} for {spec.travellers} traveller(s), {spec.constraints.pace} pace"
        f"{', interests: ' + ', '.join(stressed) if stressed else ''}{', by ' + mode if mode else ''}{assumed}."
    )
    return {"trip_spec": spec, "edit_locks": locks, "assistant_reply": None, "agent_meta": meta, "agent_messages": [message]}
