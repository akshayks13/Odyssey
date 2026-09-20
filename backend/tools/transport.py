"""Road, rail, and air quotes for a hop."""
from __future__ import annotations

from datetime import date, timedelta

from langchain_core.tools import tool

from tools.cost_calculator import check_transport_availability
from tools.mapbox_api import geocode_location, get_directions, haversine_km
from tools.travel_market import _city_iata, search_flights

_MODES = ("road", "rail", "air")


def pair_key(origin: str, destination: str) -> str:
    return f"{origin.strip()}|{destination.strip()}"


def _probe_date(iso_date: str = "") -> str:
    if iso_date:
        return iso_date
    return (date.today() + timedelta(days=21)).isoformat()


@tool(parse_docstring=True)
def quote_transport(origin: str, destination: str, mode: str = "road", date: str = "") -> dict:
    """Time, cost, and availability for one transport mode between two places.

    Args:
        origin: Origin city name.
        destination: Destination city name.
        mode: One of road, rail, air.
        date: Optional YYYY-MM-DD for flight quotes.

    Returns:
        dict with available, mode, duration_hours, cost_inr, distance_km, source, reason.
    """
    mode = (mode or "road").lower()
    if mode not in _MODES:
        mode = "road"

    availability = check_transport_availability.invoke(
        {"origin": origin, "destination": destination, "mode": mode}
    )
    if not availability.get("available"):
        return {
            "available": False,
            "mode": mode,
            "duration_hours": 0.0,
            "cost_inr": 0.0,
            "distance_km": 0.0,
            "source": "rules",
            "reason": availability.get("reason", "unavailable"),
        }

    if mode == "air":
        flight = search_flights.invoke(
            {"origin": origin, "destination": destination, "date": _probe_date(date)}
        )
        if not flight.get("available"):
            return {
                "available": False,
                "mode": "air",
                "duration_hours": 0.0,
                "cost_inr": 0.0,
                "distance_km": 0.0,
                "source": flight.get("source", "rules"),
                "reason": flight.get("reason", "no flight offers"),
                "origin_iata": flight.get("origin_iata"),
                "destination_iata": flight.get("destination_iata"),
            }
        o_geo = geocode_location.invoke({"place_name": origin})
        d_geo = geocode_location.invoke({"place_name": destination})
        distance_km = round(haversine_km(o_geo["lat"], o_geo["lng"], d_geo["lat"], d_geo["lng"]), 1)
        duration = float(flight.get("duration_hours") or 0) or round(max(1.2, distance_km / 750.0 + 0.85), 1)
        airline = flight.get("airline")
        o_iata = flight.get("origin_iata") or _city_iata(origin)
        d_iata = flight.get("destination_iata") or _city_iata(destination)
        price = float(flight.get("cheapest_price_inr") or 0)
        label = airline or "Flight"
        summary = f"{label} {o_iata}→{d_iata} · {duration:.1f}h · ₹{int(price):,}"
        return {
            "available": True,
            "mode": "air",
            "duration_hours": duration,
            "cost_inr": price,
            "distance_km": distance_km,
            "source": flight.get("source", "generated"),
            "reason": "intercity air",
            "origin_iata": o_iata,
            "destination_iata": d_iata,
            "airline": airline,
            "summary": summary,
        }

    road = get_directions.invoke({"origin": origin, "destination": destination})
    if mode == "rail":
        hours = round(float(road["duration_hours"]) * 1.2, 2)
        cost = round(float(road["cost_inr"]) * 0.45, 0)
        return {
            "available": True,
            "mode": "rail",
            "duration_hours": hours,
            "cost_inr": cost,
            "distance_km": road["distance_km"],
            "source": "rail_estimate",
            "reason": availability.get("reason", "regional rail"),
            "summary": f"Train {origin}→{destination} · {hours:.1f}h · ₹{int(cost):,}",
        }

    hours = float(road["duration_hours"])
    cost = float(road["cost_inr"])
    return {
        "available": True,
        "mode": "road",
        "duration_hours": hours,
        "cost_inr": cost,
        "distance_km": road["distance_km"],
        "source": road.get("source", "mapbox"),
        "reason": "road taxi/bus",
        "summary": f"Road {origin}→{destination} · {hours:.1f}h · ₹{int(cost):,}",
    }


def quote_all_modes(origin: str, destination: str, date: str = "") -> dict[str, dict]:
    return {
        mode: quote_transport.invoke(
            {"origin": origin, "destination": destination, "mode": mode, "date": date}
        )
        for mode in _MODES
    }


def pick_mode(origin: str, destination: str, quotes: dict[str, dict] | None = None) -> str:
    """Pick road, rail, or air from quotes and Mapbox drive time."""
    quotes = quotes or quote_all_modes(origin, destination)
    air = quotes.get("air") or {}
    rail = quotes.get("rail") or {}
    road = quotes.get("road") or {}
    road_hours = float(road.get("duration_hours") or 0)
    o_iata, d_iata = _city_iata(origin), _city_iata(destination)

    if air.get("available") and o_iata and d_iata and o_iata != d_iata:
        if road_hours >= 8.0 or road_hours < 1.0:
            return "air"
    if rail.get("available") and road_hours >= 2.0:
        return "rail"
    return "road"
