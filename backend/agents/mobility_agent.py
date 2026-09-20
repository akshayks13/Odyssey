"""Mobility — order selected cities and label each hop.

Role: mover. Visit order is weighted A* on Mapbox road times. Mode labels
(road / rail / air) sit on top of that order. Does not pick cities or hotels.

Decides: start city among selected stops; mode label per hop when the quote
is available. Honours "by train / flight / road" in the user request.

Computes: NetworkX graph + weighted A*; Mapbox road quotes; air only when both
places have IATA codes.

Uses: `quote_transport`, `search_flights`, `check_transport_availability`,
`calculate_route_cost`. Optional arrival hop from origin_city → first stop.
"""
from __future__ import annotations

from algorithms.astar import astar_route_search, build_travel_graph
from llm import get_llm, llm_decide, llm_provider_name
from models.schemas import DisruptionType, Route, RouteLeg, TransportMode
from orchestration.state import TripState
from tools.cost_calculator import calculate_route_cost, check_transport_availability
from tools.transport import pair_key, pick_mode, quote_all_modes, quote_transport
from tools.travel_market import search_flights

_ALLOWED_MODES = {m.value for m in TransportMode}


def _user_mode_preference(raw_input: str) -> str | None:
    text = (raw_input or "").lower()
    if any(w in text for w in ("by train", "prefer train", "by rail", "prefer rail")):
        return "rail"
    if any(w in text for w in ("by flight", "prefer flight", "by air", "prefer air")):
        return "air"
    if any(w in text for w in ("by car", "self drive", "by road", "by taxi", "prefer road")):
        return "road"
    return None


def _parse_modes(raw) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, mode in raw.items():
        text = str(mode).lower().strip()
        if text not in _ALLOWED_MODES:
            continue
        if "|" in str(key):
            out[str(key)] = text
        elif isinstance(key, str) and "-" in key:
            left, right = key.split("-", 1)
            out[pair_key(left.strip(), right.strip())] = text
    return out


def _leg_from_quote(origin: str, destination: str, quoted: dict, *, available: bool) -> RouteLeg:
    return RouteLeg(
        origin=origin,
        destination=destination,
        mode=TransportMode(quoted.get("mode", "road")),
        distance_km=float(quoted.get("distance_km") or 0),
        duration_hours=float(quoted.get("duration_hours") or 0),
        cost_inr=float(quoted.get("cost_inr") or 0),
        available=available,
        summary=quoted.get("summary"),
        airline=quoted.get("airline"),
        origin_iata=quoted.get("origin_iata"),
        destination_iata=quoted.get("destination_iata"),
        source=quoted.get("source"),
        reason=quoted.get("reason"),
    )


def mobility_agent_node(state: TripState) -> dict:
    selected = state.get("selected_destinations", [])
    disruptions = state.get("disruptions", [])
    spec = state.get("trip_spec")
    names = [d.name for d in selected]

    if not names:
        return {
            "route": Route(ordered_destinations=[]),
            "agent_messages": ["Mobility Agent: no destinations available to route."],
        }

    transport_disrupted = {d.target for d in disruptions if d.type == DisruptionType.TRANSPORT}
    probe_date = spec.start_date if spec and spec.start_date else ""
    preferred = _user_mode_preference(spec.raw_input if spec else "")
    origin_city = (spec.origin_city or "").strip() if spec else ""
    if origin_city.lower() in {n.lower() for n in names}:
        origin_city = ""

    start = names[0]
    llm_note = ""
    llm_modes: dict[str, str] = {}
    decision = llm_decide(
        get_llm(),
        tools=[quote_transport, search_flights, check_transport_availability, calculate_route_cost],
        system=(
            "You are Odyssey's Mobility agent — the mover, not the city picker. "
            "ROLE: help label transport between the cities Destination already chose. "
            "YOU DECIDE: a start city among those selected stops, and a mode label "
            "(road|rail|air) per hop when that quote is available. "
            "YOU MUST NOT: invent a visit order, add new cities, or pick hotels. "
            "Visit ORDER is computed by weighted A* on Mapbox road times after you reply. "
            "Call quote_transport with mode=road so those times exist. "
            "Two airport cities a full day apart by road must be air. "
            "Places without airports stay on the road network. "
            "Honour the user's preferred mode when that quote is available. "
            "Reply ONLY with JSON: "
            '{"start": "City", "modes": {"Origin|Destination": "road|rail|air"}, "reasoning": "..."}.'
        ),
        user=(
            f"Selected destinations: {names}. "
            f"Arrival from: {origin_city or 'already in region'}. "
            f"User preferred mode: {preferred or 'unspecified'}. "
            f"Transport disruptions: {sorted(transport_disrupted)}."
        ),
    )
    if decision.get("start") in names:
        start = decision["start"]
        llm_note = f" LLM start={start}"
    llm_modes = _parse_modes(decision.get("modes"))
    if llm_modes:
        llm_note += " LLM modes"
    if decision.get("_tool_calls"):
        llm_note += f" tools={decision['_tool_calls']}"

    def mapbox_road_leg(origin: str, destination: str) -> dict:
        quoted = quote_transport.invoke(
            {"origin": origin, "destination": destination, "mode": "road", "date": probe_date}
        )
        result = {**quoted, "mode": "road"}
        if origin in transport_disrupted or destination in transport_disrupted:
            result = {
                **result,
                "duration_hours": float(result.get("duration_hours") or 0) * 5,
                "cost_inr": float(result.get("cost_inr") or 0) * 3,
            }
        return result

    cap = spec.constraints.max_daily_travel_hours if spec else 4.0
    graph = build_travel_graph(names, mapbox_road_leg)
    search_result = astar_route_search(graph, start, names, max_daily_travel_hours=cap)
    order = search_result["order"]

    def labelled_hop(origin: str, destination: str) -> dict:
        quotes = quote_all_modes(origin, destination, probe_date)
        heuristic = pick_mode(origin, destination, quotes)
        mode = heuristic
        if preferred and quotes.get(preferred, {}).get("available"):
            mode = preferred
        key = pair_key(origin, destination)
        llm_mode = llm_modes.get(key)
        if llm_mode and quotes.get(llm_mode, {}).get("available"):
            if not (heuristic == "air" and llm_mode == "road" and preferred != "road"):
                mode = llm_mode
        quoted = quotes.get(mode) or quotes.get("road") or mapbox_road_leg(origin, destination)
        if not quoted.get("available"):
            quoted = quotes.get("road") or mapbox_road_leg(origin, destination)
            mode = "road"
        return {**quoted, "mode": mode}

    legs: list[RouteLeg] = []
    total_cost = 0.0
    total_distance = 0.0
    total_hours = 0.0
    road_sources: list[str] = []

    if origin_city:
        hop = labelled_hop(origin_city, order[0])
        legs.append(_leg_from_quote(origin_city, order[0], hop, available=True))
        total_cost += legs[-1].cost_inr
        total_distance += legs[-1].distance_km
        total_hours += legs[-1].duration_hours
        llm_note += f" arrive={origin_city}->{order[0]} {legs[-1].mode.value}"

    for i in range(len(order) - 1):
        hop = labelled_hop(order[i], order[i + 1])
        if hop.get("mode") == "road":
            road_sources.append(str(hop.get("source") or "mapbox"))
        legs.append(
            _leg_from_quote(
                order[i],
                order[i + 1],
                hop,
                available=order[i + 1] not in transport_disrupted,
            )
        )
        total_cost += legs[-1].cost_inr
        total_distance += legs[-1].distance_km
        total_hours += legs[-1].duration_hours

    route = Route(
        ordered_destinations=order,
        legs=legs,
        total_distance_km=round(total_distance, 1),
        total_duration_hours=round(total_hours, 2),
        total_cost_inr=round(total_cost, 2),
        search_algorithm=search_result["algorithm"],
        nodes_expanded=search_result["nodes_expanded"],
    )

    hop_bits = ", ".join(
        leg.summary or f"{leg.origin}->{leg.destination} {leg.mode.value}" for leg in legs
    ) or "single stop"
    graph_src = ", ".join(sorted(set(road_sources))) or "mapbox/seed road"
    source = f" via {llm_provider_name()} + Mapbox + A*" if llm_note else f" via Mapbox + A* ({graph_src})"
    who = f" user={preferred}" if preferred else ""
    message = (
        f"Mobility Agent: {search_result['algorithm']} ordered {' -> '.join(order)} "
        f"on road graph ({search_result['nodes_expanded']} nodes, {graph_src}); "
        f"hops: {hop_bits}{source}.{llm_note}{who}"
    )
    return {"route": route, "agent_messages": [message]}
