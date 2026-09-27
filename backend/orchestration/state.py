"""Shared LangGraph state for the six agents."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from models.schemas import (
    AgentConflict,
    BudgetBreakdown,
    Destination,
    Activity,
    Disruption,
    EditDirective,
    EditLocks,
    Hotel,
    Itinerary,
    ReplanDirective,
    Route,
    StayBlock,
    TripSpec,
    ValidationReport,
)


class TripState(TypedDict, total=False):
    # --- Input -------------------------------------------------------
    raw_input: str
    trip_spec: Optional[TripSpec]

    # --- Discovery (Destination Agent) --------------------------------
    candidate_destinations: list[Destination]
    candidate_activities: dict[str, list[Activity]]
    selected_destinations: list[Destination]

    # --- Mobility (Mobility Agent) ------------------------------------
    route: Optional[Route]

    # --- Budget (Budget Agent) ----------------------------------------
    budget_breakdown: Optional[BudgetBreakdown]
    accommodation_options: list[Hotel]
    excluded_activity_ids: list[str]
    stay_plan: list[StayBlock]

    # --- Schedule (Itinerary Architect) --------------------------------
    draft_itinerary: Optional[Itinerary]
    final_itinerary: Optional[Itinerary]

    # --- Validation (Critic & Replanner) -------------------------------
    validation_report: Optional[ValidationReport]
    replan_directives: list[ReplanDirective]
    iteration_count: int
    conflicts: list[AgentConflict]

    # --- Live / disruption events ---------------------------------------
    disruptions: list[Disruption]

    # --- Prompt-based editing (Edit Router) -------------------------------
    edit_request: Optional[str]  # a pending user edit; consumed by edit_router
    edit_directive: Optional[EditDirective]
    edit_locks: EditLocks  # standing instructions every specialist honours
    assistant_reply: Optional[str]  # answer-only turns

    # --- Streaming log (accumulates across the whole run) ---------------
    agent_messages: Annotated[list[str], operator.add]

    # How the agent that just ran reached its answer: the tools it called and the algorithms it ran.
    # The UI shows these as chips; they are evidence, not plan data.
    agent_meta: dict

    # --- Score -------------------------------------------------------
    optimization_score: float


def initial_state(raw_input: str) -> TripState:
    return TripState(
        raw_input=raw_input,
        trip_spec=None,
        candidate_destinations=[],
        candidate_activities={},
        selected_destinations=[],
        route=None,
        budget_breakdown=None,
        accommodation_options=[],
        excluded_activity_ids=[],
        stay_plan=[],
        draft_itinerary=None,
        final_itinerary=None,
        validation_report=None,
        replan_directives=[],
        iteration_count=0,
        conflicts=[],
        disruptions=[],
        edit_request=None,
        edit_directive=None,
        edit_locks=EditLocks(),
        assistant_reply=None,
        agent_messages=[],
        agent_meta={},
        optimization_score=0.0,
    )
