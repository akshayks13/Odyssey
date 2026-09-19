"""End-to-end scenario tests against the compiled LangGraph.

These cover the three demo scenarios required by the plan:
  1. Normal 5-day Kerala planning
  2. Weather/closure disruption with targeted replan
  3. Budget overrun / budget-cut disruption
"""
from __future__ import annotations

from models.schemas import Disruption, DisruptionType
from orchestration.graph import get_graph
from orchestration.state import initial_state


GRAPH = get_graph()


def _run(raw_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return GRAPH.invoke(initial_state(raw_input), config=config), config


def test_scenario_normal_kerala_trip():
    result, _ = _run(
        "I have 5 days in Kerala with 3 friends. We have around 40000 rupees. We like nature and adventure, but relaxed pace.",
        "scenario-normal",
    )
    assert result["trip_spec"].destination_region.lower().startswith("kerala")
    assert result["trip_spec"].duration_days == 5
    assert result["selected_destinations"]
    assert result["route"] is not None
    assert result["budget_breakdown"] is not None
    itinerary = result.get("final_itinerary") or result.get("draft_itinerary")
    assert itinerary is not None
    assert len(itinerary.days) == 5
    assert any(day.items for day in itinerary.days), "at least one day must have scheduled activities"
    assert result["validation_report"] is not None
    names = [m.split(":")[0] for m in result["agent_messages"]]
    for agent in ("Trip Analyst", "Destination Agent", "Mobility Agent", "Budget Agent", "Itinerary Architect", "Critic"):
        assert any(agent in n for n in names)


def test_scenario_weather_closure_replans_destination():
    result, config = _run(
        "5 day trip to Kerala for 2 people, budget 50000, love nature and adventure.",
        "scenario-closure",
    )
    original = [d.name for d in result["selected_destinations"]]
    assert original
    closed = original[0]

    disruption = Disruption(
        type=DisruptionType.CLOSURE,
        target=closed,
        description="Heavy rain damage, closed for repairs",
        day=1,
    )
    GRAPH.update_state(
        config,
        {"disruptions": [disruption], "iteration_count": 0},
        as_node="itinerary_architect",
    )
    result2 = GRAPH.invoke(None, config=config)

    new_names = [d.name for d in result2["selected_destinations"]]
    assert closed not in new_names, "closed destination must be swapped out"
    assert result2.get("final_itinerary") is not None
    # Critic must have routed through destination_agent (targeted, not a full restart)
    later_messages = result2["agent_messages"][len(result["agent_messages"]) :]
    joined = " ".join(later_messages)
    assert "destination_agent" in joined or "Destination Agent" in joined
    assert "Trip Analyst" not in joined  # analyst is NOT re-invoked on targeted replan


def test_scenario_budget_cut_triggers_budget_agent():
    result, config = _run(
        "4 day trip to Kerala for 2 people, budget 35000, relaxing beach vibes.",
        "scenario-budget-cut",
    )
    original_total = result["budget_breakdown"].total_inr
    original_ceiling = result["budget_breakdown"].ceiling_inr
    assert original_total <= original_ceiling

    disruption = Disruption(
        type=DisruptionType.BUDGET_CUT,
        target="trip",
        description="Sponsor withdrew, budget cut",
        new_budget_inr=18000,
    )
    GRAPH.update_state(
        config,
        {"disruptions": [disruption], "iteration_count": 0},
        as_node="itinerary_architect",
    )
    result2 = GRAPH.invoke(None, config=config)

    later_messages = result2["agent_messages"][len(result["agent_messages"]) :]
    joined = " ".join(later_messages)
    assert "Budget Agent" in joined or "budget_agent" in joined
    assert result2["budget_breakdown"].ceiling_inr == 18000
    # Either repaired under the new ceiling, or returned a best-effort plan with a warning
    assert result2.get("final_itinerary") is not None or result2.get("draft_itinerary") is not None
