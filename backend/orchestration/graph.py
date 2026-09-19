"""
LangGraph wiring for Odyssey.

Pipeline (locked architecture — see project plan):

    Trip Analyst -> Destination -> Mobility -> Budget -> Itinerary Architect -> Critic
                                                                                   |
                       (targeted replan: only the broken specialist re-runs) <-----+

Mobility runs before Budget (not in parallel) so the transport cost from the
chosen route is available when Budget computes the total. Running Mobility
and Budget in parallel would let Architect fire twice on the first pass and
can deadlock on a single-agent targeted replan.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.budget_agent import budget_agent_node
from agents.critic_replanner import critic_replanner_node
from agents.destination_agent import destination_agent_node
from agents.itinerary_architect import itinerary_architect_node
from agents.mobility_agent import mobility_agent_node
from agents.trip_analyst import trip_analyst_node
from orchestration.routing import route_after_critic
from orchestration.state import TripState


def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("trip_analyst", trip_analyst_node)
    graph.add_node("destination_agent", destination_agent_node)
    graph.add_node("mobility_agent", mobility_agent_node)
    graph.add_node("budget_agent", budget_agent_node)
    graph.add_node("itinerary_architect", itinerary_architect_node)
    graph.add_node("critic_replanner", critic_replanner_node)

    graph.set_entry_point("trip_analyst")
    graph.add_edge("trip_analyst", "destination_agent")
    graph.add_edge("destination_agent", "mobility_agent")
    graph.add_edge("mobility_agent", "budget_agent")
    graph.add_edge("budget_agent", "itinerary_architect")
    graph.add_edge("itinerary_architect", "critic_replanner")

    graph.add_conditional_edges(
        "critic_replanner",
        route_after_critic,
        {
            "valid": END,
            "max_iterations": END,
            "replan_destination": "destination_agent",
            "replan_budget": "budget_agent",
            "replan_mobility": "mobility_agent",
            "rebuild_schedule": "itinerary_architect",
        },
    )

    return graph.compile(checkpointer=MemorySaver())


_compiled_graph = None


def get_graph():
    """Singleton compiled graph — compiled once, reused across requests."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
