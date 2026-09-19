"""
Agent 4 — Mobility & Routing

Decides the ORDER to visit the selected destinations (weighted A* search
over the destination graph) and fetches actual road travel time/cost per
leg from Mapbox (or seed data). Also checks for injected transport
disruptions and re-routes around them when this agent is re-invoked.
"""
from __future__ import annotations

from algorithms.astar import astar_route_search, build_travel_graph
from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import DisruptionType, Route, RouteLeg, TransportMode
from orchestration.state import TripState
from tools.amadeus_api import search_flights
from tools.cost_calculator import calculate_route_cost, check_transport_availability
from tools.mapbox_api import get_directions


def mobility_agent_node(state: TripState) -> dict:
    selected = state.get("selected_destinations", [])
    disruptions = state.get("disruptions", [])
    names = [d.name for d in selected]

    if not names:
        return {
            "route": Route(ordered_destinations=[]),
            "agent_messages": ["Mobility Agent: no destinations available to route."],
        }

    transport_disrupted = {d.target for d in disruptions if d.type == DisruptionType.TRANSPORT}

    start = names[0]
    llm_note = ""
    decision = llm_decide(
        get_llm(),
        tools=[get_directions, search_flights, check_transport_availability, calculate_route_cost],
        system=(
            "You are Odyssey's Mobility agent. Call get_directions to inspect legs. "
            "search_flights is only for long-haul air, never for intra-Kerala hops. "
            "Reply ONLY with JSON: {\"start\": \"City\", \"reasoning\": \"...\"}. "
            "Start should be a gateway city already in the selected list (Kochi if present, otherwise first)."
        ),
        user=f"Selected destinations: {names}. Transport disruptions: {sorted(transport_disrupted)}.",
    )
    if decision.get("start") in names:
        start = decision["start"]
        llm_note = f" LLM start={start}"
        if decision.get("_tool_calls"):
            llm_note += f" tools={decision['_tool_calls']}"

    def get_leg(origin: str, destination: str) -> dict:
        result = get_directions.invoke({"origin": origin, "destination": destination})
        if origin in transport_disrupted or destination in transport_disrupted:
            # Simulate disruption: inflate cost/time heavily so A* naturally
            # avoids this edge if any alternative ordering exists.
            result = {**result, "duration_hours": result["duration_hours"] * 5, "cost_inr": result["cost_inr"] * 3}
        return result

    spec = state.get("trip_spec")
    cap = spec.constraints.max_daily_travel_hours if spec else 4.0
    graph = build_travel_graph(names, get_leg)
    search_result = astar_route_search(graph, start, names, max_daily_travel_hours=cap)

    order = search_result["order"]
    legs: list[RouteLeg] = []
    total_cost = 0.0
    for i in range(len(order) - 1):
        leg_data = get_leg(order[i], order[i + 1])
        legs.append(
            RouteLeg(
                origin=order[i],
                destination=order[i + 1],
                mode=TransportMode(leg_data.get("mode", "road")),
                distance_km=leg_data["distance_km"],
                duration_hours=leg_data["duration_hours"],
                cost_inr=leg_data["cost_inr"],
                available=order[i + 1] not in transport_disrupted,
            )
        )
        total_cost += leg_data["cost_inr"]

    route = Route(
        ordered_destinations=order,
        legs=legs,
        total_distance_km=search_result["total_distance_km"],
        total_duration_hours=search_result["total_time_hours"],
        total_cost_inr=total_cost,
        search_algorithm=search_result["algorithm"],
        nodes_expanded=search_result["nodes_expanded"],
    )

    source = f" via {llm_provider_name()} + A*" if llm_note else " via weighted A*"
    message = (
        f"Mobility Agent: {search_result['algorithm']} ordered route {' -> '.join(order)} "
        f"({search_result['total_time_hours']:.1f}h travel, {search_result['nodes_expanded']} nodes expanded)"
        f"{source}.{llm_note}"
    )

    return {"route": route, "agent_messages": [message]}
