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
                    not raised when Budget trimmed that city's sights, or the field report explains the gap
  ALL_TRAVEL        not a single sight fits                 -> Mobility
  BUDGET_CUT        a new, lower ceiling was reported       -> Budget
  CLOSURE/WEATHER   the traveller reported one at a city    -> Destination
  TRANSPORT         a strike at a city                      -> Mobility
  HEAVY_RAIN        the field report says heavy rain on a day that has outdoor sights   -> Itinerary Architect
  SIGHT_CLOSED      the field report says a scheduled sight is closed that day          -> Itinerary Architect
Routing: a reported event first, then by severity (high before medium). An issue that is still there
after its agent already had a retry is not sent again (a field issue is sent again only when it is a new fact).

The strategy decides what happens to an issue: "targeted" sends it to the agent above; "restart" always
sends the plan back to the Destination agent so everything re-runs; "none" only reports it.
"""
from __future__ import annotations

from agents.base import Agent, Post, Result
from config import MAX_REPLAN_ITERATIONS
from core.messages import TRAVELLER, Message, MsgType
from models.schemas import (
    AgentConflict,
    Disruption,
    DisruptionType,
    EditLocks,
    IssueSeverity,
    ObservedFacts,
    ReplanDirective,
    TransportMode,
    ValidationIssue,
    ValidationReport,
)
from tools import world
from tools.budget_validator import validate_budget
from tools.schedule_validator import check_schedule_conflicts
from tools.world import OUTDOOR_CATEGORIES

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
# Field issues are facts about specific days: a second one is a different fact, not a failed retry.
_FIELD_ISSUES = {"HEAVY_RAIN", "SIGHT_CLOSED"}


def _field_explains(day, observed: ObservedFacts, sights_by_city: dict) -> bool:
    """A free day that heavy rain or a closure explains is not a sign of too few sights."""
    if not day.date:
        return False
    if observed.is_heavy_rain(day.destination, day.date):
        return True
    return any(observed.is_closed(a.id, day.date) for a in sights_by_city.get(day.destination, []))


def _road_leg_on(itinerary, city: str, when: str) -> bool:
    return any(
        day.date == when and day.travel_leg and day.travel_leg.mode == TransportMode.ROAD and city in (day.travel_leg.origin, day.travel_leg.destination)
        for day in itinerary.days
    )


def find_issues(state: dict) -> list[ValidationIssue]:
    itinerary = state.get("draft_itinerary")
    budget = state.get("budget_breakdown")
    route = state.get("route")
    spec = state.get("trip_spec")
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    disruptions: list[Disruption] = state.get("disruptions", [])
    observed: ObservedFacts = state.get("observed") or ObservedFacts()
    sights_by_city = state.get("candidate_activities") or {}
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
        trimmed = {a.destination for acts in sights_by_city.values() for a in acts
                   if a.id in set(state.get("excluded_activity_ids") or [])}
        for day in itinerary.days:
            # A day emptied by budget trimming (or by a field report) is not a reason to change cities.
            if (day.kind == "leisure" and day.day_number not in locks.free_days and day.destination not in trimmed
                    and not _field_explains(day, observed, sights_by_city)):
                issues.append(ValidationIssue(
                    type="LEISURE_DAY", day=day.day_number, severity=IssueSeverity.MEDIUM,
                    message=f"Day {day.day_number} in {day.destination} has nothing left to see — too few sights for the days there.",
                    target_agent="destination_agent",
                ))
            if day.date:
                outdoors = [i.activity_name for i in day.items if i.kind == "activity" and i.category in OUTDOOR_CATEGORIES]
                if outdoors and observed.is_heavy_rain(day.destination, day.date):
                    issues.append(ValidationIssue(
                        type="HEAVY_RAIN", day=day.day_number, severity=IssueSeverity.MEDIUM, target_agent="itinerary_architect",
                        message=f"Heavy rain reported in {day.destination} on day {day.day_number}, but outdoor sights are scheduled: {', '.join(outdoors)}.",
                    ))
                shut = [i.activity_name for i in day.items if i.kind == "activity" and observed.is_closed(i.activity_id, day.date)]
                if shut:
                    issues.append(ValidationIssue(
                        type="SIGHT_CLOSED", day=day.day_number, severity=IssueSeverity.MEDIUM, target_agent="itinerary_architect",
                        message=f"{', '.join(shut)} reported closed on day {day.day_number}, but scheduled then.",
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
        if kind and d.date and itinerary and not _road_leg_on(itinerary, d.target, d.date):
            continue  # a strike on one day matters only if the plan drives in or out of that city that day
        if kind and d.target in touched:
            issues.append(ValidationIssue(type=kind, day=d.day, severity=IssueSeverity.HIGH,
                                          message=f"{kind} disruption at {d.target}: {d.description}",
                                          target_agent=_DISRUPTION_TARGET_AGENT[d.type]))
    return issues


class CriticAgent(Agent):
    name = "critic_replanner"
    label = "Critic & Replanner"
    role = "Coordinator: validates the draft against the rules and the field, and sends each problem to the one agent that owns it"
    architecture = "utility-based (picks the most severe issue and the cheapest agent to fix it)"
    peas = {
        "performance": "Every real problem caught; each sent to exactly one owner; the fewest replan loops; a valid plan accepted; the best plan so far returned when no retry can help",
        "environment": "The draft plan, the budget, reported events, the field report, and the other agents' declared ownership",
        "actuators": "Write the validation report; send REPLAN to one agent, or ACCEPT / BEST_EFFORT to the traveller; fold handled events into standing choices",
        "sensors": "Everything on the blackboard (read only), incoming SCHEDULE_READY, FIELD_REPORT and DISRUPTION messages",
    }
    reads = (
        "draft_itinerary", "budget_breakdown", "disruptions", "iteration_count", "trip_spec", "route", "edit_locks",
        "selected_destinations", "candidate_activities", "excluded_activity_ids", "replan_directives", "edit_directive", "conflicts", "observed",
    )
    writes = ("validation_report", "replan_directives", "iteration_count", "conflicts", "final_itinerary", "disruptions", "edit_locks", "trip_spec")

    def handle(self, msg: Message, view: dict) -> Result:
        itinerary = view["draft_itinerary"]
        budget = view["budget_breakdown"]
        disruptions: list[Disruption] = view["disruptions"]
        iteration = view["iteration_count"]

        issues = find_issues(view)
        valid = all(i.severity == IssueSeverity.LOW for i in issues)  # a LOW issue is a note (a drive kept because you asked for it), not a fault
        report = ValidationReport(valid=valid, issues=issues, score=itinerary.optimization_score if itinerary else 0.0)

        conflicts = list(view["conflicts"])
        if budget and not budget.is_within_budget and view["selected_destinations"]:
            top = max(view["selected_destinations"], key=lambda d: d.preference_score)
            conflicts.append(AgentConflict(
                subject=top.name,
                claims={"destination_agent": f"{top.name} preference score {top.preference_score:.2f}",
                        "budget_agent": f"Budget over by ₹{budget.over_budget_by_inr:,.0f}"},
                resolution="Critic routes to the cheapest-damage repair (hotel tier or travel mode) before dropping the city",
            ))

        previous = (view["replan_directives"] or [None])[-1]
        retried = previous.constraints.get("issue") if previous else None
        editing = view["edit_directive"] is not None
        actionable = [
            i for i in issues
            if i.target_agent
            and not (editing and i.type in _ONLY_WHILE_PLANNING)
            and (i.type != retried or (i.type in _FIELD_ISSUES and previous is not None and i.message != previous.reason))
        ]

        directives: list[ReplanDirective] = []
        posts: list[Post] = []
        final, remaining = None, disruptions
        if valid:
            final, remaining = itinerary, []
            message = "Critic: itinerary valid — no violations detected." if not issues else f"Critic: itinerary valid; {len(issues)} note(s) for you."
            posts.append(Post(MsgType.ACCEPT, TRAVELLER, "plan accepted"))
        elif self.strategy.replan == "none":
            final, remaining = itinerary, []
            message = f"Critic: {len(issues)} issue(s) found ({', '.join(dict.fromkeys(i.type for i in issues))}); this strategy does not replan — returning the plan as it stands."
            posts.append(Post(MsgType.BEST_EFFORT, TRAVELLER, f"{len(issues)} issue(s) left unfixed"))
        elif iteration + 1 >= MAX_REPLAN_ITERATIONS:
            final, remaining = itinerary, []
            message = f"Critic: max iterations ({MAX_REPLAN_ITERATIONS}) reached with {len(issues)} unresolved issue(s) — returning best-effort plan."
            posts.append(Post(MsgType.BEST_EFFORT, TRAVELLER, f"iteration limit, {len(issues)} issue(s) left"))
        elif not actionable:
            final, remaining = itinerary, []
            message = f"Critic: {len(issues)} issue(s) remain after a retry — returning the plan as it stands."
            posts.append(Post(MsgType.BEST_EFFORT, TRAVELLER, f"{len(issues)} issue(s) left after a retry"))
        else:
            reported = {*_DISRUPTION_ISSUE_TYPE.values(), "BUDGET_CUT"}
            top = sorted(actionable, key=lambda i: (i.type not in reported, _SEVERITY_ORDER[i.severity]))[0]
            if self.strategy.replan == "restart":  # no diagnosis: everything re-runs from the first planning agent
                target, constraints = "destination_agent", {"issue": top.type, "avoid": [], "restart": True}
            else:
                target, constraints = top.target_agent, {"issue": top.type, "avoid": top.avoid}
            directive = ReplanDirective(target_agent=target, reason=top.message, constraints=constraints)
            directives.append(directive)
            message = f"Critic: found {len(issues)} issue(s) (top: {top.type}), routing to {target} (replan iteration {iteration + 1}/{MAX_REPLAN_ITERATIONS})."
            posts.append(Post(MsgType.REPLAN, target, f"{top.type}: {top.message}"[:110], {"directive": directive}))

        updates: dict = {}
        if not remaining and disruptions:
            # Handled disruptions become standing facts, so a later edit cannot bring a closed city or the old budget back.
            locks = (view["edit_locks"] or EditLocks()).model_copy(deep=True)
            gone = [d.target for d in disruptions if d.type in (DisruptionType.CLOSURE, DisruptionType.WEATHER)]
            locks.excluded_cities = list(dict.fromkeys([*locks.excluded_cities, *gone]))
            locks.pinned_cities = [c for c in locks.pinned_cities if c not in gone]
            updates["edit_locks"] = locks
            cuts = [d.new_budget_inr for d in disruptions if d.type == DisruptionType.BUDGET_CUT and d.new_budget_inr]
            if cuts and view["trip_spec"]:
                updates["trip_spec"] = view["trip_spec"].model_copy(update={"budget_inr": min(cuts)})

        updates.update(
            validation_report=report,
            replan_directives=directives,
            iteration_count=iteration + 1,
            conflicts=conflicts,
            final_itinerary=final,
            disruptions=remaining,
        )
        return Result(
            updates=updates,
            posts=posts,
            message=message,
            tools=["validate_budget", "check_schedule_conflicts"],
            algorithms=["rule-based validation, severity-sorted routing"],
            note="valid" if valid else f"{len(issues)} issue(s): " + ", ".join(dict.fromkeys(i.type for i in issues)),
        )
