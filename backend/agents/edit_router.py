"""Edit Router — reads a change request for an existing plan and picks the agent to re-run from.

Role: front door for changes. The Critic only reacts to problems it can detect (over budget,
overlaps, a closure); "make day 2 lighter" breaks no rule, so it needs its own agent. The
router inspects the plan with tools, records what the user asked for as `EditLocks` (which every
specialist reads, so later replans keep the edit), and re-enters the graph at `route_to`.
The Critic then validates the result as usual.

Decides (LLM): what changed, whether it is only a question, and which agent to re-run.

Computes: checks names against the real plan, applies the change to the spec and locks, and
enforces one rule: the entry point may not be later than the earliest agent whose inputs
changed. Re-running more is fine; skipping an agent whose inputs moved would leave stale data.

Needs a language model; without one it leaves the plan alone and says so.
Tools: `get_plan_day`, `find_in_plan`, `list_alternative_cities`, `geocode_location`.
"""
from __future__ import annotations

import json
from datetime import date

from langchain_core.tools import tool

from algorithms.planning import PACE_CAPS
from llm import agent_trace, get_llm, llm_decide
from models.schemas import Disruption, DisruptionType, EditDirective, EditLocks, TripSpec
from orchestration.state import TripState
from tools.mapbox_api import geocode_location

# Earliest agent first. Re-running from an earlier one is always safe.
ENTRY_ORDER = ["destination_agent", "mobility_agent", "budget_agent", "itinerary_architect", "critic_replanner"]

SYSTEM_PROMPT = """You are Odyssey's Edit Router. A traveller already has a plan and wants to change it. Read their message, look at the plan (CONTEXT and tools), and decide what changed and which agent should re-run. You never build or rewrite the plan.

Choose route_to = the EARLIEST agent whose inputs the change touches:
- destination_agent: which cities/sights are offered; interest weights (nature, food...); adding/removing a city; the arrival city or region.
- mobility_agent: how they travel (road/rail/air) and what it costs; number of travellers; trip start date.
- budget_agent: hotels, trip length, budget.
- itinerary_architect: the day-by-day schedule: pace, a packed day, free days, a later start, skipping a sight, moving a sight to another day.
- critic_replanner: a real-world event (road closed, strike, storm) hit the plan.
- answer: only a question, or the request is unclear.

Reply ONLY with JSON:
{"intent": "modify" | "answer", "route_to": "<agent or answer>", "summary": "one short sentence",
 "reply": "for intent=answer: the answer from the plan, or ONE clarifying question",
 "spec_patch": {"duration_days": int, "budget_inr": number, "travellers": int, "start_date": "YYYY-MM-DD", "origin_city": str,
                "destination_region": str, "pace": "relaxed|moderate|packed", "preferences": {"nature|adventure|food|nightlife|relaxation|culture|shopping": 0..1}},
 "add_cities": [str], "remove_cities": [str],
 "lock_updates": {"preferred_mode": "road|rail|air", "hotel_prefs": {"<city lowercase or *>": "cheapest|best|<hotel name>"},
                  "excluded_activities": [str], "pinned_activities": {"<activity name>": day_number},
                  "free_days": [int], "light_days": [int], "pace": "relaxed|moderate|packed", "day_start_hour": number},
 "disruptions": [{"type": "weather|closure|transport|budget_cut", "target": "<city>", "description": str, "new_budget_inr": number}]}

Include only what the user changed; leave everything else out. Use names exactly as in CONTEXT or tool results (call find_in_plan / get_plan_day if unsure). To add a place not in the plan, check it exists with geocode_location. preferences are the new absolute weights (0-1). A disruption is a real-world event on a whole city; something they just don't want to do is a lock. Today's date is """


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


def _context(state: TripState) -> dict:
    spec: TripSpec = state["trip_spec"]
    itinerary = state.get("final_itinerary") or state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    route = state.get("route")
    return {
        "trip": {
            "region": spec.destination_region,
            "origin": spec.origin_city,
            "days": spec.duration_days,
            "travellers": spec.travellers,
            "budget_inr": spec.budget_inr,
            "start_date": spec.start_date,
            "pace": spec.constraints.pace,
            "preferences": spec.preferences.as_dict(),
        },
        "cities": _selected_names(state),
        "route": [f"{l.origin}->{l.destination} {l.mode.value} {l.duration_hours:.1f}h ₹{l.cost_inr:.0f}" for l in (route.legs if route else [])],
        "budget": (
            {
                "total": budget.total_inr,
                "ceiling": budget.ceiling_inr,
                "hotels": budget.hotels_inr,
                "food": budget.food_inr,
                "activities": budget.activities_inr,
                "transport": budget.transport_inr,
                "hotel_picks": [f"{h.destination}: {h.name} ₹{h.price_per_night_inr:.0f}/night" for h in budget.selected_hotels],
            }
            if budget
            else None
        ),
        "days": [
            {
                "day": d.day_number,
                "date": d.date,
                "city": d.destination,
                "kind": d.kind,
                "activities": [i.activity_name for i in d.items if i.kind == "activity"],
                "hotel": d.overnight_hotel.name if d.overnight_hotel else None,
            }
            for d in (itinerary.days if itinerary else [])
        ],
        "active_instructions": (state.get("edit_locks") or EditLocks()).describe(),
    }


def _make_tools(state: TripState) -> list:
    """Read-only tools over this plan, so the model can check names and days instead of guessing."""
    itinerary = state.get("final_itinerary") or state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    selected = _selected_names(state)

    @tool(parse_docstring=True)
    def get_plan_day(day_number: int) -> dict:
        """Get what is planned on one day of the current itinerary.

        Args:
            day_number: 1-based day number.

        Returns:
            dict with city, date, kind, scheduled items with times, and the overnight hotel.
        """
        day = next((d for d in (itinerary.days if itinerary else []) if d.day_number == day_number), None)
        if day is None:
            return {"found": False, "days_in_plan": len(itinerary.days) if itinerary else 0}
        return {
            "found": True,
            "city": day.destination,
            "date": day.date,
            "kind": day.kind,
            "items": [f"{i.activity_name} {i.start_hour:.1f}-{i.end_hour:.1f}" for i in day.items],
            "hotel": day.overnight_hotel.name if day.overnight_hotel else None,
        }

    @tool(parse_docstring=True)
    def find_in_plan(query: str) -> dict:
        """Find cities, activities and hotels in the current plan whose name contains the query.

        Args:
            query: Part of a city, sight or hotel name, e.g. "museum".

        Returns:
            dict with matching cities, activities (with their day, if scheduled) and hotels.
        """
        q = _norm(query)
        scheduled = {i.activity_name: d.day_number for d in (itinerary.days if itinerary else []) for i in d.items if i.kind == "activity"}
        return {
            "cities": [c for c in selected if q in _norm(c)],
            "activities": [{"name": n, "day": scheduled.get(n)} for n in _all_activity_names(state) if q in _norm(n)][:10],
            "hotels": [h.name for h in (budget.selected_hotels if budget else []) if q in _norm(h.name)],
        }

    @tool(parse_docstring=True)
    def list_alternative_cities() -> dict:
        """List other places in the region that are not in the plan yet, best match first.

        Returns:
            dict with a list of city names.
        """
        return {"cities": [d.name for d in state.get("candidate_destinations", []) if d.name not in selected][:8]}

    return [get_plan_day, find_in_plan, list_alternative_cities, geocode_location]


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
        preferred_mode=raw.get("preferred_mode") if raw.get("preferred_mode") in {"road", "rail", "air"} else None,
        hotel_prefs={str(k).lower(): str(v) for k, v in (raw.get("hotel_prefs") or {}).items() if v},
        excluded_activities=[str(x) for x in raw.get("excluded_activities") or [] if x],
        pinned_activities={str(k): int(v) for k, v in (raw.get("pinned_activities") or {}).items() if _num(v) is not None},
        free_days=[int(x) for x in raw.get("free_days") or [] if _num(x) is not None],
        light_days=[int(x) for x in raw.get("light_days") or [] if _num(x) is not None],
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
    d.add_cities = [c.strip() for c in dict.fromkeys(d.add_cities) if c.strip() and c.strip().lower() not in {s.lower() for s in selected}]
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
    for key in ("origin_city", "destination_region"):
        if patch.get(key):
            clean[key] = str(patch[key]).strip()
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

    if "destination_region" in patch:
        spec.destination_region = patch["destination_region"]
        locks.pinned_cities, locks.excluded_cities, locks.hotel_prefs = [], [], {}
        levels.append(0)
    if "origin_city" in patch and patch["origin_city"] != spec.origin_city:
        spec.origin_city = patch["origin_city"]
        levels.append(0)
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
        return {"edit_request": None, "edit_directive": directive, "assistant_reply": directive.reply, "agent_messages": [f"Edit Router: {note}"]}

    llm = get_llm()
    decision = (
        llm_decide(llm, tools=_make_tools(state), system=SYSTEM_PROMPT + date.today().isoformat() + ".",
                   user=f"CONTEXT: {json.dumps(_context(state), default=str, separators=(',', ':'))}\n\nUSER MESSAGE: {message}", max_rounds=5)
        if llm
        else {}
    )
    directive = _directive_from_json(decision) if decision else None
    if directive is None:
        return answer(
            EditDirective(reply="I couldn't reach the language model to understand that change, so your plan is unchanged. Please try again in a moment."),
            "language model unavailable — plan unchanged.",
        )

    directive, missing = _sanitize(directive, state)
    if directive.intent == "answer":
        directive.reply = directive.reply or "Tell me what you'd like to change."
        return answer(directive, "answered without changing the plan.")

    spec, locks, disruptions, floor = _apply(directive, state)
    if floor < 0:
        found = f" I couldn't find {', '.join(missing)} in your plan." if missing else ""
        directive.reply = f"Nothing needed to change.{found}"
        return answer(directive, f"no effective change.{found}")

    chosen = ENTRY_ORDER.index(directive.entry) if directive.entry in ENTRY_ORDER else floor
    entry = ENTRY_ORDER[min(chosen, floor)]  # never later than the earliest agent whose inputs changed
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
        "agent_meta": agent_trace(decision, note=f"re-running from {entry.replace('_', ' ')}"),
        "agent_messages": [f"Edit Router: {directive.summary}{found} → re-running from {entry.replace('_', ' ')}."],
    }
