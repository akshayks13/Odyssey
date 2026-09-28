"""Mobility — order the selected cities and choose how to travel each hop.

Role: mover. Visit order is weighted A* on Mapbox road times. The LLM then picks
road / rail / air for every hop (including getting there and home), using
flight and train quotes it asks for. Does not pick cities or hotels.

Decides: the mode per hop. Honours what the user asked for ("by train", "fly"). A drive longer than the
daily travel limit is replaced by the fastest flight or train that exists, unless they asked for road.
A place with no airport is flown to via the nearest one (the quote includes the ride); if both ends
use the same airport or station there is nothing to fly or ride and the hop stays on the road.

Computes: NetworkX graph + weighted A* (tried from every start city, best kept);
road quotes (Mapbox, else OSRM); group prices (seats x travellers for air/rail; for road the vehicle
the model chose: own car, taxi, tempo traveller or bus, priced per vehicle or seat by `estimate_road_cost`).

Tools the model can call: `search_public_transport`, `check_transport_disruptions`.
The code uses `get_directions` (road quotes) and `calculate_route_cost`.
"""
from __future__ import annotations

import re

from algorithms.astar import astar_route_search, build_travel_graph
from llm import agent_trace, get_llm, llm_decide
from models.schemas import DisruptionType, EditLocks, Route, RouteLeg, TransportMode
from orchestration.state import TripState
from tools.cost_calculator import ROAD_VEHICLES, calculate_route_cost, estimate_road_cost
from tools.mapbox_api import get_directions
from tools.travel_market import check_transport_disruptions, search_public_transport

_MODES = {m.value for m in TransportMode}


def _pair(origin: str, destination: str) -> str:
    return f"{origin}|{destination}"


def _leg(origin: str, destination: str, quote: dict, mode: str, travellers: int, available: bool = True, vehicle: str | None = None) -> RouteLeg:
    """Air and rail quotes are per seat. A road hop is priced for the whole group by the vehicle the model chose."""
    hours = float(quote.get("duration_hours") or 0)
    if mode == "road":
        priced = estimate_road_cost.invoke({"distance_km": float(quote.get("distance_km") or 0), "travellers": travellers, "vehicle": vehicle or "taxi"})
        vehicle = priced["vehicle"]
        cost = priced["cost_inr"] * float(quote.get("cost_multiplier") or 1.0)  # a strike makes the journey dearer
        summary = f"Road ({ROAD_VEHICLES[vehicle]['label']}) {origin}→{destination} · {hours:.1f}h · ₹{int(cost):,}"
        if vehicle == "bus" and travellers > 1:
            summary += f" for {travellers}"
    else:
        vehicle = None
        cost = float(quote.get("cost_inr") or 0) * travellers
        summary = quote.get("summary") or f"{mode.title()} {origin}→{destination} · {hours:.1f}h"
        if travellers > 1:
            summary += f" each · ₹{int(cost):,} for {travellers}"
    return RouteLeg(
        origin=origin,
        destination=destination,
        mode=TransportMode(mode),
        distance_km=float(quote.get("distance_km") or 0),
        duration_hours=float(quote.get("duration_hours") or 0),
        cost_inr=cost,
        available=available,
        summary=summary,
        airline=quote.get("operator"),
        vehicle=vehicle,
        source=quote.get("source"),
        reason=quote.get("reason"),
    )


def mobility_agent_node(state: TripState) -> dict:
    selected = state.get("selected_destinations", [])
    disruptions = state.get("disruptions", [])
    spec = state.get("trip_spec")
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    names = [d.name for d in selected]

    if not names:
        return {
            "route": Route(ordered_destinations=[]),
            "agent_messages": ["Mobility Agent: no destinations available to route."],
        }

    transport_disrupted = {d.target for d in disruptions if d.type == DisruptionType.TRANSPORT}
    travel_date = spec.start_date if spec and spec.start_date else ""
    travellers = spec.travellers if spec else 1
    origin_city = (spec.origin_city or "").strip() if spec else ""
    if origin_city.lower() in {n.lower() for n in names}:
        origin_city = ""

    def road_quote(origin: str, destination: str) -> dict:
        quote = dict(get_directions.invoke({"origin": origin, "destination": destination}))
        if origin in transport_disrupted or destination in transport_disrupted:  # a strike: slower and pricier
            quote["duration_hours"] = float(quote.get("duration_hours") or 0) * 5
            quote["cost_multiplier"] = 3.0
        return quote

    def road_costs(quote: dict) -> dict:
        """What this hop would cost the whole group in each kind of vehicle, so the model can weigh it against fares."""
        return {v: estimate_road_cost.invoke({"distance_km": float(quote.get("distance_km") or 0), "travellers": travellers, "vehicle": v})["cost_inr"] for v in ROAD_VEHICLES}

    # 1. Visit order: weighted A* over Mapbox road times, from whichever start city is quickest overall.
    cap = spec.constraints.max_daily_travel_hours if spec else 4.0
    graph = build_travel_graph(names, lambda o, d: road_quote(o, d))
    search = min(
        (astar_route_search(graph, start, names, max_daily_travel_hours=cap) for start in names),
        key=lambda r: r["total_time_hours"],
    )
    order = search["order"]

    # 2. Every hop, with its road quote. The LLM chooses the mode for each.
    hops: list[tuple[str, str]] = []
    if origin_city:
        hops.append((origin_city, order[0]))
    hops += list(zip(order, order[1:]))
    if origin_city:
        hops.append((order[-1], origin_city))
    road = {_pair(o, d): road_quote(o, d) for o, d in hops}

    sent_back = [d.reason for d in state.get("replan_directives", []) if d.target_agent == "mobility_agent"]
    decision = llm_decide(
        get_llm(),
        tools=[search_public_transport, check_transport_disruptions],
        system=(
            "You are Odyssey's Mobility agent — the mover, not the city picker. "
            "ROLE: choose how to travel each hop: road, rail or air. "
            "YOU DECIDE: the mode per hop. You know which places have airports and stations and when a "
            "flight or train is worth it over a long drive; call search_public_transport to get a quote "
            "before choosing rail or air (a place without an airport is quoted with the ride to the nearest one; "
            "if both ends would use the same airport the quote is unavailable, and that hop is by road); "
            "call check_transport_disruptions if a storm could stop a journey on the travel date. "
            "Short drives stay on the road. Honour what the traveller asked for. "
            "For road hops also choose ONE road_vehicle for the trip: own_car when they drive their own car from home "
            "(they say so, or it is a road trip from where they live), tempo_traveller for a group of about 6 to 12, "
            "bus only for a budget group on a well-served route, otherwise taxi (someone who flew in has no car). "
            "Weigh road_cost_inr_by_vehicle against the air and rail fares. "
            "Fares must leave room for hotels, food and sights: on a tight budget prefer a train over a flight. "
            "Never fix a budget by choosing a very long drive (longer than the daily travel limit): take the flight or train even if the plan then runs over budget, which is reported. "
            "YOU MUST NOT: reorder the cities, add cities, or pick hotels. "
            'Reply ONLY with JSON: {"modes": {"Origin|Destination": "road|rail|air"}, "road_vehicle": "own_car|taxi|tempo_traveller|bus", "reasoning": "..."}.'
        ),
        user=(
            f"Traveller's request: {spec.raw_input if spec else ''!r}. "
            f"Preferred mode (from later instructions): {locks.preferred_mode or 'none'}. "
            f"{travellers} traveller(s), travelling on {travel_date or 'a date to be decided'}, "
            f"with a total budget of ₹{(spec.budget_inr if spec else 0):,.0f} for everything (fares are part of it). "
            f"Hops with their road quote: {[{'hop': k, 'road_km': v['distance_km'], 'road_hours': v['duration_hours'], 'road_cost_inr_by_vehicle': road_costs(v)} for k, v in road.items()]}. "
            f"Transport disruptions: {sorted(transport_disrupted)}."
            + (f" The reviewer sent this back because: {'; '.join(sent_back)}." if sent_back else "")
        ),
    )
    modes = {
        "|".join(part.strip() for part in re.split(r"\||→|->", str(k)) if part.strip()): str(v).lower()
        for k, v in (decision.get("modes") or {}).items()
        if str(v).lower() in _MODES
    }
    if locks.preferred_mode in _MODES:
        modes = {k: locks.preferred_mode for k in road}
    vehicle = str(decision.get("road_vehicle") or "").lower()
    if vehicle not in ROAD_VEHICLES:  # the model named none: a traveller who flies or takes a train in has no car
        vehicle = "taxi" if any(m in ("air", "rail") for m in modes.values()) else "own_car"

    # 3. Build the legs. A drive longer than the daily travel limit is not recommended unless the traveller asked for
    # road: look for a flight or train and take the fastest. If no quote is available, stay on the road.
    def make_leg(origin: str, destination: str) -> RouteLeg:
        pair = _pair(origin, destination)
        mode = modes.get(pair, "road")
        available = destination not in transport_disrupted
        road_hours = float(road[pair]["duration_hours"])
        too_long = mode == "road" and road_hours > cap and locks.preferred_mode != "road"
        options = ["air", "rail"] if too_long else ([mode] if mode in ("air", "rail") else [])
        quotes = []
        for option in options:
            quote = search_public_transport.invoke({"origin": origin, "destination": destination, "mode": option, "date": travel_date})
            if quote.get("available") and (not too_long or float(quote["duration_hours"]) <= 0.6 * road_hours):  # materially faster, or the drive stays
                quotes.append((option, quote))
        if quotes:
            option, quote = min(quotes, key=lambda oq: float(oq[1]["duration_hours"])) if too_long else quotes[0]
            return _leg(origin, destination, quote, option, travellers, available)
        return _leg(origin, destination, road[pair], "road", travellers, available, vehicle)

    legs = [make_leg(o, d) for o, d in hops]
    return_leg = legs.pop() if origin_city else None

    total = [*legs, *([return_leg] if return_leg else [])]
    route = Route(
        ordered_destinations=order,
        legs=legs,
        total_distance_km=round(sum(l.distance_km for l in total), 1),
        total_duration_hours=round(sum(l.duration_hours for l in total), 2),
        total_cost_inr=round(calculate_route_cost.invoke({"leg_costs_inr": [l.cost_inr for l in total], "travellers": travellers})["total_inr"], 2),
        search_algorithm=search["algorithm"],
        nodes_expanded=search["nodes_expanded"],
        return_leg=return_leg,
    )

    hop_bits = ", ".join(l.summary or f"{l.origin}->{l.destination} {l.mode.value}" for l in total) or "single stop"
    message = (
        f"Mobility Agent: {search['algorithm']} ordered {' -> '.join(order)} on road graph "
        f"({search['nodes_expanded']} nodes); hops: {hop_bits}."
    )
    meta = agent_trace(
        decision,
        algorithms=[f"{search['algorithm']} ({search['nodes_expanded']} nodes expanded)"],
        note="; ".join(
            [f"{pair.replace('|', ' → ')}: {mode}" for pair, mode in modes.items()]
            + ([f"road by {ROAD_VEHICLES[vehicle]['label']}"] if any(l.mode.value == "road" for l in total) else [])
        ),
    )
    return {"route": route, "agent_meta": meta, "agent_messages": [message]}
