"""Critic & Replanner — validate the draft and route one specialist if invalid.

Role: coordinator. Does not rewrite the itinerary itself. Max 3 replan loops.
Never restarts Trip Analyst.

Decides: which specialist to re-invoke (destination / mobility / budget /
architect) for the cheapest-damage repair.

Computes: budget, schedule, weather, closure, and transport checks; severity
sort. On the last allowed pass, returns a best-effort plan with warnings.

Uses: `validate_budget`, `check_schedule_conflicts`, `check_transport_disruptions`,
`check_attraction_availability`, `check_weather_disruptions`. The JSON reply is
the replan directive — there is no separate generate_replan_directive tool.
"""
from __future__ import annotations

from config import MAX_REPLAN_ITERATIONS
from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import (
    AgentConflict,
    Disruption,
    DisruptionType,
    IssueSeverity,
    ReplanDirective,
    ValidationIssue,
    ValidationReport,
)
from orchestration.state import TripState
from tools.travel_market import check_transport_disruptions
from tools.budget_validator import validate_budget
from tools.foursquare_api import check_attraction_availability
from tools.schedule_validator import check_schedule_conflicts
from tools.weather import check_weather_disruptions

_SEVERITY_ORDER = {IssueSeverity.HIGH: 0, IssueSeverity.MEDIUM: 1, IssueSeverity.LOW: 2}

_DISRUPTION_TARGET_AGENT = {
    DisruptionType.WEATHER: "destination_agent",
    DisruptionType.CLOSURE: "destination_agent",
    DisruptionType.TRANSPORT: "mobility_agent",
    DisruptionType.BUDGET_CUT: "budget_agent",
}

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

    active_cuts = [d.new_budget_inr for d in disruptions if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
    effective_ceiling = min(active_cuts) if active_cuts else (budget.ceiling_inr if budget else None)

    if budget and effective_ceiling is not None:
        b_check = validate_budget.invoke({"total_inr": budget.total_inr, "ceiling_inr": effective_ceiling})
        if not b_check["within_budget"]:
            issues.append(
                ValidationIssue(
                    type="BUDGET",
                    severity=IssueSeverity.HIGH,
                    message=f"Over budget by ₹{b_check['over_by_inr']:.0f}",
                    target_agent="budget_agent",
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
    else:
        issues_sorted = sorted(issues, key=lambda i: _SEVERITY_ORDER.get(i.severity, 1))
        top_issue = issues_sorted[0]
        target = top_issue.target_agent or "itinerary_architect"
        llm_note = ""
        decision = llm_decide(
            get_llm(),
            tools=[validate_budget, check_schedule_conflicts, check_transport_disruptions, check_attraction_availability, check_weather_disruptions],
            system=(
                "You are Odyssey's Critic & Replanner — the coordinator, not a seventh planner. "
                "ROLE: validate the draft and, if it fails, re-invoke exactly one specialist. "
                "YOU DECIDE: target_agent = destination_agent | mobility_agent | budget_agent | "
                "itinerary_architect. "
                "YOU MUST NOT: rewrite days yourself, restart Trip Analyst, or re-run every agent. "
                "Call validator tools if you need more evidence. Pick the cheapest-damage repair. "
                "Reply ONLY with JSON: {\"target_agent\": \"destination_agent|budget_agent|"
                "mobility_agent|itinerary_architect\", \"reason\": \"...\"}."
            ),
            user=f"Issues: {[i.model_dump() for i in issues]}. Heuristic top: {top_issue.type} -> {target}.",
        )
        allowed = {"destination_agent", "budget_agent", "mobility_agent", "itinerary_architect"}
        if decision.get("target_agent") in allowed:
            target = decision["target_agent"]
            llm_note = f" LLM routed to {target}"
            if decision.get("_tool_calls"):
                llm_note += f" tools={decision['_tool_calls']}"
        replan_directives.append(
            ReplanDirective(
                target_agent=target,
                reason=decision.get("reason") or top_issue.message,
                constraints={},
            )
        )

        remaining_disruptions = disruptions
        source = f" via {llm_provider_name()}" if llm_note else ""
        message = (
            f"Critic: found {len(issues)} issue(s) (top: {top_issue.type}), routing to "
            f"{target}{source} (replan iteration {iteration + 1}/{MAX_REPLAN_ITERATIONS}).{llm_note}"
        )

    return {
        "validation_report": report,
        "replan_directives": replan_directives,
        "iteration_count": iteration + 1,
        "conflicts": conflicts,
        "final_itinerary": final_itinerary,
        "disruptions": remaining_disruptions,
        "agent_messages": [message],
    }
