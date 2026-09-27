"""Critic & Replanner — validate the draft and send it back to exactly one specialist if it fails.

Role: coordinator. Never rewrites the plan itself and never restarts the Trip Analyst. At most
MAX_REPLAN_ITERATIONS loops; after that the best plan so far is returned with its issues listed.

Checks -> who fixes it:
  BUDGET            over the ceiling                       -> Budget (one tier cheaper), or Mobility
                                                              (cheapest modes) when fares are >= 40% of it
  SCHEDULE_CONFLICT overlapping items or a > 14h day        -> Itinerary Architect
  TRAVEL_OVERLOAD   a hop longer than the daily limit       -> Destination, told to avoid the far city
                    (unless the traveller named both cities: then it is only reported)
  LEISURE_DAY       a day with nothing left to see          -> Destination (pick cities with more to do);
                    not raised when Budget trimmed that city's sights
  ALL_TRAVEL        not a single sight fits                 -> Mobility
  BUDGET_CUT        a new, lower ceiling was reported       -> Budget
  CLOSURE/WEATHER   the traveller reported one at a city    -> Destination
  TRANSPORT         a strike at a city                      -> Mobility
Routing: a reported disruption first, then by severity (high before medium). An issue that is still there
after its agent already had a retry is not sent again.
"""
from __future__ import annotations

from agents.common import trace
from config import MAX_REPLAN_ITERATIONS
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
from tools.budget_validator import validate_budget
from tools import world
from tools.schedule_validator import check_schedule_conflicts

_SEVERITY_ORDER = {IssueSeverity.HIGH: 0, IssueSeverity.MEDIUM: 1, IssueSeverity.LOW: 2}

_DISRUPTION_TARGET_AGENT = {
    DisruptionType.WEATHER: "destination_agent",
    DisruptionType.CLOSURE: "destination_agent",
    DisruptionType.TRANSPORT: "mobility_agent",
}
_DISRUPTION_ISSUE_TYPE = {
    DisruptionType.WEATHER: "WEATHER",
    DisruptionType.CLOSURE: "CLOSURE",
    DisruptionType.TRANSPORT: "TRANSPORT",
}
# Plan-quality problems worth re-picking cities for while a plan is first built. An edit is not a reason
# to re-pick them: the traveller asked for one change.
_ONLY_WHILE_PLANNING = {"LEISURE_DAY", "TRAVEL_OVERLOAD", "ALL_TRAVEL"}


def find_issues(state: TripState) -> list[ValidationIssue]:
    itinerary = state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    route = state.get("route")
    spec = state.get("trip_spec")
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    disruptions: list[Disruption] = state.get("disruptions", [])
    issues: list[ValidationIssue] = []

    cuts = [d.new_budget_inr for d in disruptions if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
    ceiling = min(cuts) if cuts else (budget.ceiling_inr if budget else None)
    if budget and cuts and budget.ceiling_inr != min(cuts):
        issues.append(ValidationIssue(type="BUDGET_CUT", severity=IssueSeverity.HIGH, target_agent="budget_agent",
                                      message=f"Budget cut to ₹{min(cuts):,.0f}: re-price the trip against it."))
    elif budget and ceiling is not None:
        check = validate_budget(budget.total_inr, ceiling)
        if not check["within_budget"]:
            fares = route.total_cost_inr if route else 0.0
            fare_heavy = fares >= 0.4 * ceiling
            issues.append(ValidationIssue(
                type="BUDGET",
                severity=IssueSeverity.HIGH,
                message=f"Over budget by ₹{check['over_by_inr']:,.0f}" + (f"; fares alone are ₹{fares:,.0f}, so use cheaper modes" if fare_heavy else ""),
                target_agent="mobility_agent" if fare_heavy else "budget_agent",
            ))

    if itinerary:
        for issue in check_schedule_conflicts([d.model_dump() for d in itinerary.days])["issues"]:
            issues.append(ValidationIssue(type="SCHEDULE_CONFLICT", day=issue["day"], severity=IssueSeverity.MEDIUM,
                                          message=issue["message"], target_agent="itinerary_architect"))
        trimmed = {a.destination for acts in (state.get("candidate_activities") or {}).values() for a in acts
                   if a.id in set(state.get("excluded_activity_ids") or [])}
        for day in itinerary.days:
            # A day emptied by budget trimming is the price of the budget, not a reason to change cities.
            if day.kind == "leisure" and day.day_number not in locks.free_days and day.destination not in trimmed:
                issues.append(ValidationIssue(
                    type="LEISURE_DAY", day=day.day_number, severity=IssueSeverity.MEDIUM,
                    message=f"Day {day.day_number} in {day.destination} has nothing left to see — too few sights for the days there.",
                    target_agent="destination_agent",
                ))
        if not any(i.kind == "activity" for d in itinerary.days for i in d.items):
            issues.append(ValidationIssue(type="ALL_TRAVEL", severity=IssueSeverity.MEDIUM,
                                          message="Travel takes all the time: not a single sight fits.", target_agent="mobility_agent"))

    if route and spec:
        named = set(locks.pinned_cities) | set(world.cities_named_in(spec.destination_region, spec.raw_input))
        for leg in route.legs:
            if leg.duration_hours > spec.constraints.max_daily_travel_hours:
                far = leg.origin if leg.destination in named else leg.destination
                asked_for = far in named  # both ends were the traveller's choice: report it, don't overrule them
                issues.append(ValidationIssue(
                    type="TRAVEL_OVERLOAD", severity=IssueSeverity.LOW if asked_for else IssueSeverity.MEDIUM,
                    message=f"{leg.origin} → {leg.destination} takes {leg.duration_hours:.1f}h by {leg.mode.value}, over the {spec.constraints.max_daily_travel_hours:g}h daily limit"
                    + (" — kept, because you asked for both." if asked_for else "."),
                    target_agent=None if asked_for else "destination_agent",
                    avoid=[] if asked_for else [far],
                ))

    touched = {d.name for d in state.get("selected_destinations", [])} | ({d.destination for d in itinerary.days} if itinerary else set())
    for d in disruptions:
        kind = _DISRUPTION_ISSUE_TYPE.get(d.type)
        if kind and d.target in touched:
            issues.append(ValidationIssue(type=kind, day=d.day, severity=IssueSeverity.HIGH,
                                          message=f"{kind} disruption at {d.target}: {d.description}",
                                          target_agent=_DISRUPTION_TARGET_AGENT[d.type]))
    return issues


def critic_replanner_node(state: TripState) -> dict:
    itinerary = state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    disruptions: list[Disruption] = state.get("disruptions", [])
    iteration = state.get("iteration_count", 0)

    issues = find_issues(state)
    valid = not issues
    report = ValidationReport(valid=valid, issues=issues, score=itinerary.optimization_score if itinerary else 0.0)

    conflicts = list(state.get("conflicts", []))
    if budget and not budget.is_within_budget and state.get("selected_destinations"):
        top = max(state["selected_destinations"], key=lambda d: d.preference_score)
        conflicts.append(AgentConflict(
            subject=top.name,
            claims={"destination_agent": f"{top.name} preference score {top.preference_score:.2f}",
                    "budget_agent": f"Budget over by ₹{budget.over_budget_by_inr:,.0f}"},
            resolution="Critic routes to the cheapest-damage repair (hotel tier or travel mode) before dropping the city",
        ))

    previous = (state.get("replan_directives") or [None])[-1]
    retried = previous.constraints.get("issue") if previous else None
    editing = state.get("edit_directive") is not None
    actionable = [i for i in issues if i.target_agent and i.type != retried and not (editing and i.type in _ONLY_WHILE_PLANNING)]

    directives: list[ReplanDirective] = []
    final, remaining = None, disruptions
    if valid:
        final, remaining = itinerary, []
        message = "Critic: itinerary valid — no violations detected."
    elif iteration + 1 >= MAX_REPLAN_ITERATIONS:
        final, remaining = itinerary, []
        message = f"Critic: max iterations ({MAX_REPLAN_ITERATIONS}) reached with {len(issues)} unresolved issue(s) — returning best-effort plan."
    elif not actionable:
        final, remaining = itinerary, []
        message = f"Critic: {len(issues)} issue(s) remain after a retry — returning the plan as it stands."
    else:
        reported = {*_DISRUPTION_ISSUE_TYPE.values(), "BUDGET_CUT"}
        top = sorted(actionable, key=lambda i: (i.type not in reported, _SEVERITY_ORDER[i.severity]))[0]
        target = top.target_agent
        directives.append(ReplanDirective(target_agent=target, reason=top.message, constraints={"issue": top.type, "avoid": top.avoid}))
        message = f"Critic: found {len(issues)} issue(s) (top: {top.type}), routing to {target} (replan iteration {iteration + 1}/{MAX_REPLAN_ITERATIONS})."

    out: dict = {}
    if not remaining and disruptions:
        # Handled disruptions become standing facts, so a later edit cannot bring a closed city or the old budget back.
        locks = (state.get("edit_locks") or EditLocks()).model_copy(deep=True)
        gone = [d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)]
        locks.excluded_cities = list(dict.fromkeys([*locks.excluded_cities, *gone]))
        locks.pinned_cities = [c for c in locks.pinned_cities if c not in gone]
        out["edit_locks"] = locks
        cuts = [d.new_budget_inr for d in disruptions if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
        if cuts and state.get("trip_spec"):
            out["trip_spec"] = state["trip_spec"].model_copy(update={"budget_inr": min(cuts)})

    return {
        **out,
        "validation_report": report,
        "replan_directives": directives,
        "iteration_count": iteration + 1,
        "conflicts": conflicts,
        "final_itinerary": final,
        "disruptions": remaining,
        "edit_request": None,
        "agent_meta": trace(
            tools=["validate_budget", "check_schedule_conflicts"],
            algorithms=["rule-based validation, severity-sorted routing"],
            note="valid" if valid else f"{len(issues)} issue(s): " + ", ".join(dict.fromkeys(i.type for i in issues)),
        ),
        "agent_messages": [message],
    }
