"""Critic & Replanner — validate the draft and route one specialist if invalid.

Role: coordinator. Does not rewrite the itinerary itself. Max 3 replan loops.
Never restarts Trip Analyst.

Decides: which specialist to re-invoke (destination / mobility / budget /
architect) for the cheapest-damage repair.

Computes: budget, schedule, empty-day, long-transfer, weather, closure, and transport
checks; severity sort. A day with nothing left to see, a hop that is too long, or a hop with no route goes
back to Destination with the reason (a city with no hotel after a retry too, told not to pick it again).
A trip that is all travel, a long road trip to get there, or a budget mostly spent on fares goes to Mobility. An issue that is still there after the agent it
went to already retried is not sent again. On the last allowed pass, returns a
best-effort plan.

Tools the model can call: `check_weather_disruptions`, `check_transport_disruptions`, to verify
a reported event. The code uses `validate_budget` and `check_schedule_conflicts`. The JSON
reply is the replan directive.
"""
from __future__ import annotations

from config import MAX_REPLAN_ITERATIONS
from llm import agent_trace, get_llm, llm_decide
from models.schemas import (
    AgentConflict,
    Disruption,
    DisruptionType,
    EditLocks,
    IssueSeverity,
    ReplanDirective,
    ValidationIssue,
    ValidationReport,
)
from orchestration.state import TripState
from tools.travel_market import check_transport_disruptions
from tools.budget_validator import validate_budget
from tools.schedule_validator import check_schedule_conflicts
from tools.weather import check_weather_disruptions

_SEVERITY_ORDER = {IssueSeverity.HIGH: 0, IssueSeverity.MEDIUM: 1, IssueSeverity.LOW: 2}

_DISRUPTION_TARGET_AGENT = {
    DisruptionType.WEATHER: "destination_agent",
    DisruptionType.CLOSURE: "destination_agent",
    DisruptionType.TRANSPORT: "mobility_agent",
    DisruptionType.BUDGET_CUT: "budget_agent",
}

# Plan-quality problems worth another look at the cities while a plan is first built. An edit is not a
# reason to re-pick them: the traveller asked for one change.
_ONLY_WHILE_PLANNING = {"LEISURE_DAY", "TRAVEL_OVERLOAD", "ALL_TRAVEL", "NO_BASE"}

_DISRUPTION_ISSUE_TYPE = {
    DisruptionType.WEATHER: "WEATHER",
    DisruptionType.CLOSURE: "CLOSURE",
    DisruptionType.TRANSPORT: "TRANSPORT",
}


def critic_replanner_node(state: TripState) -> dict:
    itinerary = state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    disruptions: list[Disruption] = state.get("disruptions", [])
    iteration = state.get("iteration_count", 0)

    touched_names = {d.name for d in state.get("selected_destinations", [])}
    if itinerary:
        touched_names |= {day.destination for day in itinerary.days}

    issues: list[ValidationIssue] = []
    decision: dict = {}  # only set when the plan fails and a specialist has to be chosen

    active_cuts = [d.new_budget_inr for d in disruptions if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
    effective_ceiling = min(active_cuts) if active_cuts else (budget.ceiling_inr if budget else None)

    if budget and effective_ceiling is not None:
        b_check = validate_budget.invoke({"total_inr": budget.total_inr, "ceiling_inr": effective_ceiling})
        if not b_check["within_budget"]:
            route_cost = state["route"].total_cost_inr if state.get("route") else 0.0
            transport_heavy = route_cost >= 0.4 * effective_ceiling  # hotels and food cannot be cut enough: change how they travel
            issues.append(
                ValidationIssue(
                    type="BUDGET",
                    severity=IssueSeverity.HIGH,
                    message=f"Over budget by ₹{b_check['over_by_inr']:.0f}"
                    + (f"; transport alone is ₹{route_cost:.0f} of the ₹{effective_ceiling:.0f} budget, so use cheaper modes" if transport_heavy else ""),
                    target_agent="mobility_agent" if transport_heavy else "budget_agent",
                )
            )

    if itinerary:
        sched_check = check_schedule_conflicts.invoke({"days": [d.model_dump() for d in itinerary.days]})
        for issue in sched_check["issues"]:
            issues.append(
                ValidationIssue(
                    type="SCHEDULE_CONFLICT",
                    day=issue["day"],
                    severity=IssueSeverity.MEDIUM,
                    message=issue["message"],
                    target_agent="itinerary_architect",
                )
            )

    locks: EditLocks = state.get("edit_locks") or EditLocks()
    if itinerary:  # a night with no hotel means the hotel lookup failed; Budget retries it
        already_retried = bool((state.get("replan_directives") or [None])[-1] and state["replan_directives"][-1].constraints.get("issue") == "NO_HOTEL")
        for city in dict.fromkeys(d.destination for d in itinerary.days[:-1] if d.overnight_hotel is None):
            if already_retried:  # Budget looked again and still found none: stay somewhere else nearby
                issues.append(
                    ValidationIssue(
                        type="NO_BASE",
                        severity=IssueSeverity.HIGH,
                        message=f"No hotel could be found in {city}, even after looking again: choose a nearby place with hotels as the base instead.",
                        target_agent="destination_agent",
                        avoid=[city],
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        type="NO_HOTEL",
                        severity=IssueSeverity.HIGH,
                        message=f"No hotel found for {city}.",
                        target_agent="budget_agent",
                    )
                )
    spec = state.get("trip_spec")
    route = state.get("route")
    if itinerary:
        for day in itinerary.days:
            if day.kind == "leisure" and day.day_number not in locks.free_days:
                issues.append(
                    ValidationIssue(
                        type="LEISURE_DAY",
                        day=day.day_number,
                        severity=IssueSeverity.MEDIUM,
                        message=f"Day {day.day_number} in {day.destination} has nothing left to see — too few sights for the days there.",
                        target_agent="destination_agent",
                    )
                )
    if itinerary and not any(i.kind == "activity" for d in itinerary.days for i in d.items):
        issues.append(
            ValidationIssue(
                type="ALL_TRAVEL",
                severity=IssueSeverity.MEDIUM,
                message="Getting there and back takes all the time: not a single sight fits. Add days, or choose a faster way to travel.",
                target_agent="mobility_agent",
            )
        )
    if route and spec:
        origin = (spec.origin_city or "").strip().lower()
        for leg in [route.legs[0] if route.legs else None, route.return_leg]:
            if leg and origin and origin in (leg.origin.lower(), leg.destination.lower()) and leg.mode.value == "road" \
                    and leg.duration_hours > spec.constraints.max_daily_travel_hours and locks.preferred_mode != "road":
                issues.append(
                    ValidationIssue(
                        type="LONG_ROAD_TRIP",
                        severity=IssueSeverity.MEDIUM,
                        message=f"{leg.origin} → {leg.destination} is {leg.duration_hours:.0f}h by road; a flight or train may be better.",
                        target_agent="mobility_agent",
                    )
                )
        for leg in [*route.legs, route.return_leg]:
            if leg and leg.mode.value == "road" and leg.source == "unavailable":  # no road between them, and no flight or train was found either
                between_stops = leg.origin.strip().lower() != origin and leg.destination.strip().lower() != origin
                issues.append(
                    ValidationIssue(
                        type="NO_ROUTE",
                        severity=IssueSeverity.HIGH,
                        message=f"No route was found between {leg.origin} and {leg.destination}: pick stops that can be reached from each other.",
                        target_agent="destination_agent",
                        avoid=[leg.destination] if between_stops else [],
                    )
                )
                break
        for leg in route.legs:
            if leg.origin.strip().lower() != origin and leg.mode.value != "air" and leg.duration_hours > spec.constraints.max_daily_travel_hours:
                issues.append(
                    ValidationIssue(
                        type="TRAVEL_OVERLOAD",
                        severity=IssueSeverity.MEDIUM,
                        message=(
                            f"{leg.origin} → {leg.destination} takes {leg.duration_hours:.1f}h by {leg.mode.value}, "
                            f"over the {spec.constraints.max_daily_travel_hours:.0f}h a day limit."
                        ),
                        target_agent="destination_agent",
                    )
                )

    for d in disruptions:
        issue_type = _DISRUPTION_ISSUE_TYPE.get(d.type)
        if issue_type is None:
            continue
        if d.target not in touched_names:
            continue
        issues.append(
            ValidationIssue(
                type=issue_type,
                day=d.day,
                severity=IssueSeverity.HIGH,
                message=f"{issue_type} disruption at {d.target}: {d.description}",
                target_agent=_DISRUPTION_TARGET_AGENT[d.type],
            )
        )

    valid = len(issues) == 0
    score = itinerary.optimization_score if itinerary else 0.0
    report = ValidationReport(valid=valid, issues=issues, score=score)

    conflicts = list(state.get("conflicts", []))
    if budget and not budget.is_within_budget and state.get("selected_destinations"):
        top_dest = max(state["selected_destinations"], key=lambda d: d.preference_score, default=None)
        if top_dest:
            conflicts.append(
                AgentConflict(
                    subject=top_dest.name,
                    claims={
                        "destination_agent": f"{top_dest.name} preference score {top_dest.preference_score:.2f}",
                        "budget_agent": f"Budget over by ₹{budget.over_budget_by_inr:.0f}",
                    },
                    resolution="Critic routes to budget_agent for cheapest-damage repair (hotel/activity trade-off)",
                )
            )

    replan_directives: list[ReplanDirective] = []
    final_itinerary = None
    remaining_disruptions = disruptions

    is_last_allowed_pass = (iteration + 1) >= MAX_REPLAN_ITERATIONS

    # Whatever was sent back last time already had its retry; if it is still here, another pass won't fix it.
    previous = (state.get("replan_directives") or [None])[-1]
    retried = previous.constraints.get("issue") if previous else None
    editing = state.get("edit_directive") is not None
    actionable = [i for i in issues if i.type != retried and not (editing and i.type in _ONLY_WHILE_PLANNING)]

    if valid:
        final_itinerary = itinerary
        remaining_disruptions = []
        message = "Critic: itinerary valid — no violations detected."
    elif is_last_allowed_pass:
        final_itinerary = itinerary  # best-effort plan after exhausting replan budget
        remaining_disruptions = []
        message = (
            f"Critic: max iterations ({MAX_REPLAN_ITERATIONS}) reached with {len(issues)} "
            "unresolved issue(s) — returning best-effort plan."
        )
    elif not actionable:
        final_itinerary = itinerary
        remaining_disruptions = []
        message = f"Critic: {len(issues)} issue(s) remain after a retry — returning the plan as it stands."
    else:
        # What the traveller reported (a closure, a storm, a strike) comes first; then by severity.
        issues_sorted = sorted(actionable, key=lambda i: (i.type not in _DISRUPTION_ISSUE_TYPE.values(), _SEVERITY_ORDER.get(i.severity, 1)))
        top_issue = issues_sorted[0]
        target = top_issue.target_agent or "itinerary_architect"
        llm_note = ""
        decision = llm_decide(
            get_llm(),
            tools=[check_weather_disruptions, check_transport_disruptions],
            system=(
                "You are Odyssey's Critic & Replanner — the coordinator, not a seventh planner. "
                "ROLE: validate the draft and, if it fails, re-invoke exactly one specialist. "
                "YOU DECIDE: target_agent = destination_agent | mobility_agent | budget_agent | "
                "itinerary_architect. "
                "YOU MUST NOT: rewrite days yourself, restart Trip Analyst, or re-run every agent. "
                "Call the weather or transport checks to verify a reported disruption. Pick the cheapest-damage repair. "
                "Reply ONLY with JSON: {\"target_agent\": \"destination_agent|budget_agent|"
                "mobility_agent|itinerary_architect\", \"reason\": \"...\"}."
            ),
            user=f"Issues: {[i.model_dump() for i in issues]}. Heuristic top: {top_issue.type} -> {target}.",
        )
        allowed = {"destination_agent", "budget_agent", "mobility_agent", "itinerary_architect"}
        if decision.get("target_agent") in allowed:
            target = decision["target_agent"]
            llm_note = f" LLM routed to {target}"
        replan_directives.append(
            ReplanDirective(
                target_agent=target,
                reason=f"{top_issue.message} {decision.get('reason') or ''}".strip(),
                constraints={"issue": top_issue.type, "avoid": top_issue.avoid},
            )
        )

        remaining_disruptions = disruptions
        message = (
            f"Critic: found {len(issues)} issue(s) (top: {top_issue.type}), routing to "
            f"{target} (replan iteration {iteration + 1}/{MAX_REPLAN_ITERATIONS}).{llm_note}"
        )

    return {
        "validation_report": report,
        "replan_directives": replan_directives,
        "iteration_count": iteration + 1,
        "conflicts": conflicts,
        "final_itinerary": final_itinerary,
        "disruptions": remaining_disruptions,
        "edit_request": None,
        "agent_meta": agent_trace(
            decision,
            algorithms=["budget / schedule / route validation"],
            note=("valid" if valid else f"{len(issues)} issue(s): " + ", ".join(dict.fromkeys(i.type for i in issues))),
        ),
        "agent_messages": [message],
    }
