"""Edit Router — reads a change request for an existing plan and picks the agent to re-run from.

Role: front door for changes. The Critic only reacts to problems it can detect (over budget, overlaps, a
closure); "make day 2 lighter" breaks no rule, so it needs its own agent. The router records what the
traveller asked for as `EditLocks` (which every specialist reads, so later replans keep the edit) and
re-enters the graph at the earliest agent whose inputs changed. The Critic then validates as usual.

How: keyword and pattern rules (no language model). It understands:
    days          "make day 2 lighter", "day 3 free", "put the houseboat on day 2"
    pace / start  "more relaxed", "packed", "start at 10", "later mornings"
    cities        "add Varkala", "remove Munnar", "swap Munnar for Vagamon"
    sights        "skip the museum"
    travel        "by train", "by road"
    hotels        "cheapest hotels", "best hotel in Munnar"
    trip          "make it 6 days", "one more day", "budget ₹60,000", "for 3 people"
    interests     "more food", "less adventure"
    events        "Munnar is closed", "storm in Alleppey", "strike in Kochi"
    questions     "how much does it cost?", "which hotels?", "what's on day 2?"
Anything else gets a short reply listing what it can change; the plan is left alone.
"""
from __future__ import annotations

import re
from datetime import date

from agents.common import trace
from agents.trip_analyst import THEME_WORDS, parse_budget, parse_days, parse_travellers
from algorithms.planning import PACE_CAPS
from models.schemas import Disruption, DisruptionType, EditDirective, EditLocks, TripSpec
from orchestration.state import TripState
from tools import world

# Earliest agent first. Re-running from an earlier one is always safe.
ENTRY_ORDER = ["destination_agent", "mobility_agent", "budget_agent", "itinerary_architect", "critic_replanner"]

HELP = (
    "I can change: a day (\"make day 2 lighter\", \"day 3 free\"), the pace or start time, cities "
    "(\"add Varkala\", \"swap Munnar for Vagamon\"), sights (\"skip the museum\"), travel (\"by train\"), "
    "hotels (\"cheapest hotels\"), the trip (\"6 days\", \"budget ₹60,000\", \"for 3 people\"), "
    "interests (\"more food\"), or report an event (\"Munnar is closed\")."
)
_QUESTION_START = ("what", "which", "how", "where", "when", "why", "is ", "are ", "does ", "do ", "can ", "tell me", "show me")
_REMOVE_WORDS = r"(remove|drop|skip|avoid|exclude|without|cancel|no more|not)"
_ADD_WORDS = r"(add|include|also visit|also go to|visit|extend to)"


def _selected_names(state: TripState) -> list[str]:
    return [d.name for d in state.get("selected_destinations", [])]


def _all_activity_names(state: TripState) -> list[str]:
    return [a.name for acts in (state.get("candidate_activities") or {}).values() for a in acts]


def _norm(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def _match(name: str, options: list[str]) -> str | None:
    """`name` as one of `options`: exact, then containment either way."""
    n = _norm(name)
    if not n:
        return None
    for o in options:
        if _norm(o) == n:
            return o
    for o in options:
        if n in _norm(o) or _norm(o) in n:
            return o
    return None


def _days_in(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\bday\s*(\d+)\b", text)]


def _sight_after(text: str, verb: str, names: list[str]) -> list[str]:
    """Sights named after a verb: "skip the tea museum and the dam" -> ["Tea Museum", "Mattupetty Dam..."]."""
    out = []
    for m in re.finditer(rf"\b{verb}\b\s+(?:the\s+)?([a-z ,'&-]+)", text):
        for part in re.split(r",|\band\b", m.group(1)):
            phrase = re.sub(r"\b(the|please|on day \d+|visit|trip)\b", " ", part).strip()
            hit = _match(phrase, names) if len(phrase) >= 3 else None
            if hit and hit not in out:
                out.append(hit)
    return out


def _answer(text: str, state: TripState) -> str:
    itinerary = state.get("final_itinerary") or state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    route = state.get("route")
    days = _days_in(text)
    if days and itinerary:
        day = next((d for d in itinerary.days if d.day_number == days[0]), None)
        if day is None:
            return f"The plan has {len(itinerary.days)} days."
        items = ", ".join(f"{i.activity_name} {int(i.start_hour):02d}:{int(round(i.start_hour % 1 * 60)):02d}" for i in day.items) or "nothing scheduled"
        hotel = f" Night at {day.overnight_hotel.name}." if day.overnight_hotel else ""
        return f"Day {day.day_number} in {day.destination} ({day.kind}): {items}.{hotel}"
    if re.search(r"\b(cost|budget|price|spend|total|expensive|cheap)\b", text) and budget:
        return (f"Total ₹{budget.total_inr:,.0f} of your ₹{budget.ceiling_inr:,.0f} budget: hotels ₹{budget.hotels_inr:,.0f}, "
                f"food ₹{budget.food_inr:,.0f}, activities ₹{budget.activities_inr:,.0f}, transport ₹{budget.transport_inr:,.0f}.")
    if re.search(r"\b(hotels?|stay|staying|sleep)\b", text) and budget:
        return "Hotels: " + "; ".join(f"{h.destination}: {h.name} (₹{h.price_per_night_inr:,.0f}/night)" for h in budget.selected_hotels) + "."
    if re.search(r"\b(weather|rain|rainy)\b", text) and itinerary:
        rainy = [d.day_number for d in itinerary.days if d.weather and d.weather.rainy]
        return f"Rain is likely on day(s) {', '.join(map(str, rainy))}; those days put indoor sights first." if rainy else "No day in the plan is likely to be rainy."
    if re.search(r"\b(route|travel|train|road|order|drive|get)\b", text) and route:
        return "Route: " + "; ".join(l.summary or f"{l.origin}→{l.destination}" for l in route.legs) + "." if route.legs else "It's a single-city trip."
    cities = ", ".join(_selected_names(state))
    return f"Your {len(itinerary.days) if itinerary else 0}-day plan covers {cities}. {HELP}"


def parse_edit(message: str, state: TripState) -> dict:
    """The change a message asks for, in the shape the rest of the router applies ({} = not understood)."""
    text = message.lower().replace("’", "'").strip()
    spec: TripSpec = state["trip_spec"]
    region = spec.destination_region
    if text.endswith("?") or text.startswith(_QUESTION_START):
        return {"intent": "answer", "route_to": "answer", "reply": _answer(text, state), "summary": "Answered a question."}

    activities = _all_activity_names(state)
    patch: dict = {}
    locks: dict = {}
    add, remove, events, said = [], [], [], []
    days = _days_in(text)

    # Events at a city: a closure, a storm, a strike.
    for kind, pattern, label in (
        ("closure", r"\b(closed|shut|closure)\b", "closed"),
        ("weather", r"\b(storm|cyclone|flood|floods|landslide)\b", "severe weather"),
        ("transport", r"\b(strike|hartal|bandh)\b", "transport strike"),
    ):
        if re.search(pattern, text):
            for c in world.cities_named_in(region, text) or _selected_names(state)[:1]:
                events.append({"type": kind, "target": c, "description": f"Reported by you: {label}"})
                said.append(f"{label} at {c}")
    if events:
        return {"intent": "modify", "route_to": "", "disruptions": events, "summary": "; ".join(said)}

    # Cities: swap / add / remove.
    named = world.cities_named_in(region, text)
    swap = re.search(r"\b(?:swap|replace|switch)\b(.+?)\b(?:for|with|to)\b(.+)", text)
    instead = re.search(r"(.+)\binstead of\b(.+)", text)
    if swap or instead:
        old_part, new_part = (swap.group(1), swap.group(2)) if swap else (instead.group(2), instead.group(1))
        remove += world.cities_named_in(region, old_part)
        add += world.cities_named_in(region, new_part)
    elif named and re.search(rf"\b{_REMOVE_WORDS}\b", text):
        remove += named
    elif named and re.search(rf"\b{_ADD_WORDS}\b", text):
        add += named
    if add:
        said.append("add " + ", ".join(add))
    if remove:
        said.append("remove " + ", ".join(remove))

    # Sights: skip, or pin to a day.
    skipped = [] if named else _sight_after(text, _REMOVE_WORDS, activities)
    if skipped:
        locks["excluded_activities"] = skipped
        said.append("skip " + ", ".join(skipped))
    pin = re.search(r"\b(?:put|move|do|schedule|keep)\b(.+?)\bon day\s*(\d+)", text)
    if pin and (hit := _match(re.sub(r"\bthe\b", " ", pin.group(1)).strip(), activities)):
        locks["pinned_activities"] = {hit: int(pin.group(2))}
        said.append(f"{hit} on day {pin.group(2)}")

    # Days: lighter, free.
    if days and re.search(r"\b(free|off|rest|nothing)\b", text):
        locks["free_days"] = days
        said.append(f"day {', '.join(map(str, days))} free")
    elif days and re.search(r"\b(light|lighter|less|too (packed|much|busy|full)|easier|relax)\b", text):
        locks["light_days"] = days
        said.append(f"day {', '.join(map(str, days))} lighter")

    # Pace and start time (for the whole trip, when no day was named).
    if not days:
        if re.search(r"\b(more relaxed|relaxed|slower|slow down|less packed|fewer sights)\b", text):
            locks["pace"] = "relaxed"
            said.append("relaxed pace")
        elif re.search(r"\b(packed|busier|more sights|see more|faster)\b", text):
            locks["pace"] = "packed"
            said.append("packed pace")
    start = re.search(r"\bstart (?:at|from|after)\s*(\d{1,2})", text)
    if start:
        locks["day_start_hour"] = float(start.group(1))
        said.append(f"start at {start.group(1)}:00")
    elif re.search(r"\b(later start|start later|sleep in|late mornings?|slow mornings?)\b", text):
        locks["day_start_hour"] = 10.0
        said.append("start at 10:00")

    # Travel mode and hotels.
    if re.search(r"\b(by train|train|rail)\b", text):
        locks["preferred_mode"] = "rail"
        said.append("travel by train")
    elif re.search(r"\b(by road|by car|by taxi|drive|driving)\b", text):
        locks["preferred_mode"] = "road"
        said.append("travel by road")
    if re.search(r"\bhotels?|stays?\b", text):
        pref = "cheapest" if re.search(r"\b(cheap|cheaper|cheapest|budget)\b", text) else "best" if re.search(r"\b(best|better|luxury|nicer|premium)\b", text) else None
        if pref:
            key = (named[0] if named else "*").lower()
            locks["hotel_prefs"] = {key: pref}
            said.append(f"{pref} hotel{'s' if key == '*' else ' in ' + named[0]}")

    # Trip length, budget, group size.
    n_days = parse_days(re.sub(r"\bday\s*\d+\b", " ", text))
    if n_days:
        patch["duration_days"] = n_days
    elif re.search(r"\b(one|a|1) (more|extra) day\b|\badd a day\b", text):
        patch["duration_days"] = spec.duration_days + 1
    elif re.search(r"\b(one|a|1) day (less|fewer|shorter)\b|\bremove a day\b|\bshorter\b", text):
        patch["duration_days"] = max(1, spec.duration_days - 1)
    if "duration_days" in patch:
        said.append(f"{patch['duration_days']} days")
    if re.search(r"\b(budget|₹|rs|inr|lakh|\d+k)\b|₹", text):
        value = parse_budget(text, spec.travellers)
        if value:
            patch["budget_inr"] = value
            said.append(f"budget ₹{value:,.0f}")
    if re.search(r"\b(people|persons|travellers|travelers|of us|adults|friends)\b", text):
        people = parse_travellers(text)
        if people:
            patch["travellers"] = people
            said.append(f"{people} traveller(s)")

    # Interests.
    prefs = {}
    for direction, weight in (("more", 0.9), ("less", 0.2)):
        for m in re.finditer(rf"\b{direction}\s+([a-z]+)", text):
            theme = next((t for t, words in THEME_WORDS.items() if m.group(1) in words or m.group(1) == t), None)
            if theme:
                prefs[theme] = weight
                said.append(f"{direction} {theme}")
    if prefs:
        patch["preferences"] = prefs

    if not said:
        if pin or re.search(rf"\b{_REMOVE_WORDS}\b", text):
            return {"intent": "answer", "route_to": "answer", "summary": "Nothing matched.",
                    "reply": "I couldn't find that city or sight in your plan. Cities: " + ", ".join(_selected_names(state)) + "."}
        return {}
    return {
        "intent": "modify",
        "route_to": "",
        "summary": "; ".join(said).capitalize() + ".",
        "spec_patch": patch,
        "add_cities": add,
        "remove_cities": remove,
        "lock_updates": locks,
    }


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _directive_from_json(data: dict) -> EditDirective | None:
    if data.get("intent") not in {"modify", "answer"}:
        return None
    raw = data.get("lock_updates") or {}
    locks = EditLocks(
        preferred_mode=raw.get("preferred_mode") if raw.get("preferred_mode") in {"road", "rail"} else None,
        hotel_prefs={str(k).lower(): str(v) for k, v in (raw.get("hotel_prefs") or {}).items() if v},
        excluded_activities=[str(x) for x in raw.get("excluded_activities") or [] if x],
        pinned_activities={str(k): int(_num(v)) for k, v in (raw.get("pinned_activities") or {}).items() if _num(v) is not None},
        free_days=[int(_num(x)) for x in raw.get("free_days") or [] if _num(x) is not None],
        light_days=[int(_num(x)) for x in raw.get("light_days") or [] if _num(x) is not None],
        pace=raw.get("pace") if raw.get("pace") in PACE_CAPS else None,
        day_start_hour=_num(raw.get("day_start_hour")),
    )
    disruptions = []
    for row in data.get("disruptions") or []:
        try:
            disruptions.append(
                Disruption(
                    type=DisruptionType(str(row.get("type"))),
                    target=str(row.get("target") or "trip"),
                    description=str(row.get("description") or "Reported by traveller"),
                    new_budget_inr=_num(row.get("new_budget_inr")),
                )
            )
        except (ValueError, TypeError):
            continue
    return EditDirective(
        intent=data["intent"],
        summary=str(data.get("summary") or "")[:200],
        reply=str(data["reply"]) if data.get("reply") else None,
        spec_patch=dict(data.get("spec_patch") or {}),
        add_cities=[str(x) for x in data.get("add_cities") or [] if x],
        remove_cities=[str(x) for x in data.get("remove_cities") or [] if x],
        lock_updates=locks,
        disruptions=disruptions,
        entry=str(data.get("route_to") or ""),
    )


def _sanitize(d: EditDirective, state: TripState) -> tuple[EditDirective, list[str]]:
    """Drop names that aren't in the plan. Returns the directive and the names it couldn't find."""
    spec: TripSpec = state["trip_spec"]
    selected, activities = _selected_names(state), _all_activity_names(state)
    missing: list[str] = []
    lu = d.lock_updates

    def real(names: list[str], options: list[str]) -> list[str]:
        out = []
        for n in names:
            hit = _match(n, options)
            (out if hit else missing).append(hit or n)
        return list(dict.fromkeys(out))

    d.remove_cities = real(d.remove_cities, selected)
    known = [world.match_city(spec.destination_region, c) for c in dict.fromkeys(d.add_cities)]
    missing += [c for c, k in zip(dict.fromkeys(d.add_cities), known) if k is None]
    d.add_cities = [k for k in known if k and k not in selected]
    lu.excluded_activities = real(lu.excluded_activities, activities)

    horizon = max(spec.duration_days, int(_num(d.spec_patch.get("duration_days")) or 0))
    lu.pinned_activities = {a: day for n, day in lu.pinned_activities.items() if (a := _match(n, activities)) and 1 <= day <= horizon}
    lu.free_days = sorted({x for x in lu.free_days if 1 <= x <= horizon})
    lu.light_days = sorted({x for x in lu.light_days if 1 <= x <= horizon})

    prefs = {}
    for key, value in lu.hotel_prefs.items():
        city = "*" if key in {"*", "all", "everywhere"} else _match(key, selected)
        if city:
            prefs[city.lower()] = value
        else:
            missing.append(key)
    lu.hotel_prefs = prefs

    kept = []
    for x in d.disruptions:
        city = _match(x.target, selected)
        if x.type == DisruptionType.BUDGET_CUT or city:
            kept.append(x.model_copy(update={"target": city or x.target}))
        else:
            missing.append(x.target)
    d.disruptions = kept

    patch, clean = d.spec_patch, {}
    if (n := _num(patch.get("duration_days"))) and 1 <= int(n) <= 30:
        clean["duration_days"] = int(n)
    if (n := _num(patch.get("budget_inr"))) and n > 0:
        clean["budget_inr"] = float(n)
    if (n := _num(patch.get("travellers"))) and int(n) >= 1:
        clean["travellers"] = int(n)
    try:
        if patch.get("start_date") and date.fromisoformat(str(patch["start_date"])) >= date.today():
            clean["start_date"] = str(patch["start_date"])
    except ValueError:
        pass
    if patch.get("pace") in PACE_CAPS:
        lu.pace = lu.pace or patch["pace"]
    weights = {k: round(min(1.0, max(0.0, float(v))), 2) for k, v in (patch.get("preferences") or {}).items() if k in spec.preferences.as_dict() and _num(v) is not None}
    if weights:
        clean["preferences"] = weights
    d.spec_patch = clean
    d.summary = d.summary or "Updated your plan."
    return d, list(dict.fromkeys(missing))


def _merge_locks(old: EditLocks, new: EditLocks) -> EditLocks:
    merged = old.model_copy(deep=True)
    for field in ("preferred_mode", "pace", "day_start_hour"):
        if getattr(new, field) is not None:
            setattr(merged, field, getattr(new, field))
    merged.excluded_activities = list(dict.fromkeys([*merged.excluded_activities, *new.excluded_activities]))
    merged.free_days = sorted({*merged.free_days, *new.free_days})
    merged.light_days = sorted({*merged.light_days, *new.light_days})
    merged.hotel_prefs = {**merged.hotel_prefs, **new.hotel_prefs}
    merged.pinned_activities = {**merged.pinned_activities, **new.pinned_activities}
    return merged


def _apply(d: EditDirective, state: TripState) -> tuple[TripSpec, EditLocks, list[Disruption], int]:
    """Apply the change to the spec and locks. Also returns the earliest agent (index in ENTRY_ORDER)
    whose inputs changed, or -1 if nothing did."""
    spec: TripSpec = state["trip_spec"].model_copy(deep=True)
    locks = _merge_locks(state.get("edit_locks") or EditLocks(), d.lock_updates)
    selected = _selected_names(state)
    levels: list[int] = []
    patch, lu = d.spec_patch, d.lock_updates

    if "preferences" in patch:
        spec.preferences = spec.preferences.model_copy(update=patch["preferences"])
        if not locks.pinned_cities:
            locks.pinned_cities = list(selected)  # a tweak to tastes should not swap out the cities
        levels.append(0)
    # A longer trip needs more sights to fill it (and may take another city), a shorter one fewer, so the
    # length goes all the way back to Destination. Budget and group size do not change what there is to see.
    for field, level in (("duration_days", 0), ("budget_inr", 2), ("travellers", 1), ("start_date", 1)):
        if field in patch and patch[field] != getattr(spec, field):
            setattr(spec, field, patch[field])
            levels.append(level)
    spec.needs_clarification = [n for n in spec.needs_clarification if not any(n.startswith(k) and f in patch for k, f in (("duration", "duration_days"), ("budget", "budget_inr"), ("travellers", "travellers"), ("start_date", "start_date")))]

    if d.add_cities or d.remove_cities:
        pinned = [c for c in (locks.pinned_cities or selected) if c not in d.remove_cities]
        pinned += [c for c in d.add_cities if c.lower() not in {p.lower() for p in pinned}]
        locks.pinned_cities = pinned
        locks.excluded_cities = [c for c in dict.fromkeys([*locks.excluded_cities, *d.remove_cities]) if c.lower() not in {a.lower() for a in d.add_cities}]
        spec.constraints = spec.constraints.model_copy(update={"max_destinations": max(spec.constraints.max_destinations, len(pinned))})
        levels.append(0)

    if lu.preferred_mode:
        levels.append(1)
    if lu.hotel_prefs:
        levels.append(2)
    if lu.excluded_activities or lu.pinned_activities or lu.free_days or lu.light_days or lu.pace or lu.day_start_hour is not None:
        levels.append(3)
    if "pace" in patch:
        spec.constraints = spec.constraints.model_copy(update={"pace": patch["pace"]})

    disruptions = list(state.get("disruptions", [])) + d.disruptions
    if d.disruptions:
        levels.append(4)
    return spec, locks, disruptions, (min(levels) if levels else -1)


def edit_router_node(state: TripState) -> dict:
    message = (state.get("edit_request") or "").strip()[:1000]
    if not message or state.get("trip_spec") is None:
        return {"edit_request": None, "assistant_reply": "There's no plan yet to edit — describe a trip first."}

    def answer(directive: EditDirective, note: str) -> dict:
        directive.intent, directive.entry = "answer", "answer"
        return {
            "edit_request": None,
            "edit_directive": directive,
            "assistant_reply": directive.reply,
            "agent_meta": trace(algorithms=["keyword / pattern rules"], note="plan unchanged"),
            "agent_messages": [f"Edit Router: {note}"],
        }

    parsed = parse_edit(message, state)
    directive = _directive_from_json(parsed) if parsed else None
    if directive is None:
        return answer(EditDirective(reply=f"I didn't recognise a change in that. {HELP}"), "no change recognised — plan unchanged.")

    directive, missing = _sanitize(directive, state)
    if directive.intent == "answer":
        return answer(directive, "answered without changing the plan.")

    spec, locks, disruptions, floor = _apply(directive, state)
    if floor < 0:
        found = f" I couldn't find {', '.join(missing)} in your plan." if missing else ""
        directive.reply = f"Nothing needed to change.{found}"
        return answer(directive, f"no effective change.{found}")

    entry = ENTRY_ORDER[floor]  # the earliest agent whose inputs changed
    directive.entry = entry
    found = f" (couldn't find {', '.join(missing)})" if missing else ""
    return {
        "trip_spec": spec,
        "edit_locks": locks,
        "edit_directive": directive,
        "disruptions": disruptions,
        "iteration_count": 0,
        "replan_directives": [],
        "final_itinerary": None,
        "edit_request": None,
        "assistant_reply": None,
        "agent_meta": trace(algorithms=["keyword / pattern rules"], note=f"re-running from {entry.replace('_', ' ')}"),
        "agent_messages": [f"Edit Router: {directive.summary}{found} → re-running from {entry.replace('_', ' ')}."],
    }
