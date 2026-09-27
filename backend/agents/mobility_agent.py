"""Mobility & Routing — choose road or rail for each pair of cities, then the visiting order.

Role: mover. Does not add or remove cities, or pick hotels.

Steps:
  1. For every pair of chosen cities, UCS over the road network (hours, km, towns passed) and, when both
     have a station, UCS over the rail network (hours, fare).
  2. Mode per pair, by rule:
       - the traveller asked for rail or road             -> that mode, when it exists
       - the Critic sent the plan back as too expensive   -> the cheaper mode
       - otherwise rail if it is no slower than the road, or if the drive is longer than the daily
         travel limit; else road (door to door)
     A transport strike at a city doubles road time and adds 50% to road cost for trips in or out of it.
  3. Visiting order: A* over (current city, cities visited) with an MST heuristic (admissible), on the
     chosen travel times. UCS (h = 0) and greedy nearest-neighbour run on the same problem as baselines.
"""
from __future__ import annotations

from agents.common import trace
from algorithms.astar import astar_order, greedy_order
from models.schemas import DisruptionType, EditLocks, Route, RouteLeg, TransportMode
from orchestration.state import TripState
from tools import world
from tools.cost_calculator import ROAD_VEHICLES, estimate_road_cost, road_vehicle_for

STATION_TRANSFER_HOURS = 0.5  # getting to and from the station
STRIKE_TIME_FACTOR = 2.0
STRIKE_COST_FACTOR = 1.5


def hop_options(region: str, a: str, b: str, travellers: int, strike: bool) -> dict[str, dict]:
    """Every way to get from a to b: {"road": {...}, "rail": {...}} (rail only when both have a station)."""
    options = {}
    road = world.road_route(region, a, b)
    if road["found"]:
        vehicle = road_vehicle_for(travellers)
        hours = road["hours"] * (STRIKE_TIME_FACTOR if strike else 1.0)
        cost = estimate_road_cost(road["km"], travellers, vehicle) * (STRIKE_COST_FACTOR if strike else 1.0)
        via = [t for t in road["path"][1:-1]]
        options["road"] = {"hours": round(hours, 2), "km": road["km"], "cost": round(cost, 2), "vehicle": vehicle, "via": via, "strike": strike}
    rail = world.rail_route(region, a, b)
    if rail["found"]:
        options["rail"] = {"hours": round(rail["hours"] + STATION_TRANSFER_HOURS, 2), "km": road["km"] if road["found"] else 0.0,
                           "cost": round(rail["fare_inr"] * travellers, 2), "vehicle": None, "via": rail["path"][1:-1], "strike": False}
    return options


def choose_mode(options: dict[str, dict], preferred: str | None, cheapest: bool, cap: float) -> str:
    if preferred in options:
        return preferred
    if cheapest:
        return min(options, key=lambda m: (options[m]["cost"], m))
    if "rail" in options and "road" in options:
        road, rail = options["road"], options["rail"]
        return "rail" if rail["hours"] <= road["hours"] or road["hours"] > cap else "road"
    return next(iter(options))


def make_leg(a: str, b: str, mode: str, opt: dict, travellers: int) -> RouteLeg:
    via = f" via {', '.join(opt['via'])}" if opt["via"] else ""
    if mode == "road":
        label = ROAD_VEHICLES[opt["vehicle"]]["label"]
        summary = f"Road ({label}) {a}→{b}{via} · {opt['hours']:.1f}h · ₹{opt['cost']:,.0f}"
        if opt["strike"]:
            summary += " · slowed by a strike"
    else:
        summary = f"Train {a}→{b}{via} · {opt['hours']:.1f}h incl. station transfers · ₹{opt['cost']:,.0f} for {travellers}"
    return RouteLeg(
        origin=a,
        destination=b,
        mode=TransportMode(mode),
        distance_km=opt["km"],
        duration_hours=opt["hours"],
        cost_inr=opt["cost"],
        summary=summary,
        vehicle=opt["vehicle"],
        source="dataset",
    )


def mobility_agent_node(state: TripState) -> dict:
    spec = state["trip_spec"]
    locks: EditLocks = state.get("edit_locks") or EditLocks()
    names = [d.name for d in state.get("selected_destinations", [])]
    region = spec.destination_region
    if not names:
        return {"route": Route(ordered_destinations=[]), "agent_messages": ["Mobility Agent: no destinations to route."]}

    strikes = {d.target for d in state.get("disruptions", []) if d.type == DisruptionType.TRANSPORT}
    cheapest = any(d.target_agent == "mobility_agent" and d.constraints.get("issue") == "BUDGET" for d in state.get("replan_directives", []))
    cap = spec.constraints.max_daily_travel_hours

    # 1-2. Every pair: options by UCS, then a mode by rule. The chosen hours become A*'s step costs.
    chosen: dict[tuple[str, str], tuple[str, dict]] = {}
    unavailable: list[str] = []  # pairs with no way to travel by the mode the traveller asked for
    dist: dict[str, dict[str, float]] = {a: {} for a in names}
    for a in names:
        for b in names:
            if a == b:
                continue
            options = hop_options(region, a, b, spec.travellers, strike=a in strikes or b in strikes)
            if not options:
                continue
            mode = choose_mode(options, locks.preferred_mode, cheapest, cap)
            if locks.preferred_mode and locks.preferred_mode not in options and a < b:
                unavailable.append(f"{a}–{b}")
            chosen[(a, b)] = (mode, options[mode])
            dist[a][b] = options[mode]["hours"]

    # 3. Order: A* with the MST heuristic; UCS and greedy on the same problem for comparison.
    search = astar_order(names, dist, "mst")
    ucs = astar_order(names, dist, "zero")
    greedy = greedy_order(names, dist)
    order = search["order"]

    legs = [make_leg(a, b, *chosen[(a, b)], spec.travellers) for a, b in zip(order, order[1:]) if (a, b) in chosen]
    route = Route(
        ordered_destinations=order,
        legs=legs,
        total_distance_km=round(sum(l.distance_km for l in legs), 1),
        total_duration_hours=round(sum(l.duration_hours for l in legs), 2),
        total_cost_inr=round(sum(l.cost_inr for l in legs), 2),
        search_algorithm=search["algorithm"],
        nodes_expanded=search["nodes_expanded"],
    )

    why = []
    if locks.preferred_mode:
        why.append(f"{locks.preferred_mode} where possible, as asked")
    used = {(l.origin, l.destination) for l in legs} | {(l.destination, l.origin) for l in legs}
    missing = [p for p in unavailable if tuple(p.split("–")) in used]
    if missing:
        why.append(f"no {locks.preferred_mode} for {', '.join(missing)}, so by {'road' if locks.preferred_mode == 'rail' else 'rail'}")
    if cheapest:
        why.append("cheapest mode per hop (sent back as over budget)")
    if strikes:
        why.append(f"strike at {', '.join(sorted(strikes))}")
    hops = "; ".join(l.summary for l in legs) or "a single stop, no travel between cities"
    message = f"Mobility Agent: {' → '.join(order)} ({route.total_duration_hours:.1f}h of travel). {hops}." + (f" No {locks.preferred_mode} for {', '.join(missing)}." if missing else "")
    return {
        "route": route,
        "agent_meta": trace(
            tools=["road_route", "rail_route"],
            algorithms=[
                "UCS on road and rail networks",
                f"{search['algorithm']}: {search['nodes_expanded']} nodes expanded",
            ],
            note=(
                f"Same problem — UCS: {ucs['nodes_expanded']} nodes, {ucs['total_hours']:.1f}h; "
                f"greedy: {greedy['total_hours']:.1f}h; A*: {search['total_hours']:.1f}h (optimal)"
                + (f". Rules: {', '.join(why)}" if why else "")
            ),
        ),
        "agent_messages": [message],
    }
