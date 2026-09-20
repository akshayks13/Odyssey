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
    Hotel,
    Itinerary,
    ReplanDirective,
    Route,
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

    # --- Streaming log (accumulates across the whole run) ---------------
    agent_messages: Annotated[list[str], operator.add]

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
        draft_itinerary=None,
        final_itinerary=None,
        validation_report=None,
        replan_directives=[],
        iteration_count=0,
        conflicts=[],
        disruptions=[],
        agent_messages=[],
        optimization_score=0.0,
    )
