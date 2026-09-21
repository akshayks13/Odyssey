"""Conditional edge logic for the Critic's targeted-replan routing."""
from __future__ import annotations

from config import MAX_REPLAN_ITERATIONS
from orchestration.state import TripState

_TARGET_TO_EDGE = {
    "destination_agent": "replan_destination",
    "budget_agent": "replan_budget",
    "mobility_agent": "replan_mobility",
    "itinerary_architect": "rebuild_schedule",
}


def route_after_critic(state: TripState) -> str:
    report = state.get("validation_report")
    iteration = state.get("iteration_count", 0)

    if report is None or report.valid:
        return "valid"

    if iteration >= MAX_REPLAN_ITERATIONS:
        return "max_iterations"

    directives = state.get("replan_directives", [])
    if not directives:
        return "valid"

    return _TARGET_TO_EDGE.get(directives[0].target_agent, "rebuild_schedule")


_EDIT_ENTRY_TO_EDGE = {
    "destination_agent": "replan_destination",
    "mobility_agent": "replan_mobility",
    "budget_agent": "replan_budget",
    "itinerary_architect": "rebuild_schedule",
    "critic_replanner": "revalidate",
}


def route_entry(state: TripState) -> str:
    """A run starts at the Trip Analyst (new plan) or the Edit Router (a user edit)."""
    return "revise" if state.get("edit_request") else "plan"


def route_after_edit(state: TripState) -> str:
    directive = state.get("edit_directive")
    if directive is None or directive.intent == "answer":
        return "answer"
    return _EDIT_ENTRY_TO_EDGE.get(directive.entry, "answer")


def route_after_analyst(state: TripState) -> str:
    spec = state.get("trip_spec")
    return "ask" if spec is None or not spec.destination_region else "go"
