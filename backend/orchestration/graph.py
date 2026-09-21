"""LangGraph pipeline: Analyst → Destination → Mobility → Budget → Architect → Critic.

A plan starts at the Analyst. A later user edit starts at the Edit Router, which re-enters the
pipeline at the shallowest agent the change touches; the Critic validates every path.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.budget_agent import budget_agent_node
from agents.critic_replanner import critic_replanner_node
from agents.destination_agent import destination_agent_node
from agents.edit_router import edit_router_node
from agents.itinerary_architect import itinerary_architect_node
from agents.mobility_agent import mobility_agent_node
from agents.trip_analyst import trip_analyst_node
from orchestration.routing import route_after_analyst, route_after_critic, route_after_edit, route_entry
from orchestration.state import TripState


def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("trip_analyst", trip_analyst_node)
    graph.add_node("destination_agent", destination_agent_node)
    graph.add_node("mobility_agent", mobility_agent_node)
    graph.add_node("budget_agent", budget_agent_node)
    graph.add_node("itinerary_architect", itinerary_architect_node)
    graph.add_node("critic_replanner", critic_replanner_node)
    graph.add_node("edit_router", edit_router_node)

    graph.add_conditional_edges(
        START,
        route_entry,
        {"plan": "trip_analyst", "revise": "edit_router"},
    )
    graph.add_conditional_edges(
        "edit_router",
        route_after_edit,
        {
            "replan_destination": "destination_agent",
            "replan_mobility": "mobility_agent",
            "replan_budget": "budget_agent",
            "rebuild_schedule": "itinerary_architect",
            "revalidate": "critic_replanner",
            "answer": END,
        },
    )
    graph.add_conditional_edges(
        "trip_analyst", route_after_analyst, {"go": "destination_agent", "ask": END}
    )
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
