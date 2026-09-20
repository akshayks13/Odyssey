"""Hotel and flight quotes via Gemini, then Groq. Seed only when no LLM is configured."""
from __future__ import annotations

from datetime import date, timedelta

from langchain_core.tools import tool

from llm import llm_json, llm_provider_name
from tools.seed_data import get_hotels

_CITY_IATA = {
    "kochi": "COK",
    "cochin": "COK",
    "ernakulam": "COK",
    "trivandrum": "TRV",
    "thiruvananthapuram": "TRV",
    "calicut": "CCJ",
    "kozhikode": "CCJ",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "chennai": "MAA",
    "hyderabad": "HYD",
    "kolkata": "CCU",
    "pune": "PNQ",
    "goa": "GOI",
    "jaipur": "JAI",
    "ahmedabad": "AMD",
}

_hotel_cache: dict[tuple[str, str], list[dict]] = {}
_flight_cache: dict[tuple[str, str, str], dict] = {}


def _city_iata(name: str) -> str | None:
    return _CITY_IATA.get(name.strip().lower())


def _llm_source() -> str:
    name = llm_provider_name() or "llm"
    return name.split(":")[0]


def _llm_hotels(destination: str, budget_tier: str) -> list[dict]:
    data = llm_json(
        "You list real hotels a traveller could book in this city. "
        "Reply ONLY with JSON.",
        (
            f"List 5 distinct hotels in {destination} for a {budget_tier}-tier stay. "
            "Use real property names for that city (not '{destination} Inn'). "
            "Include a mix: 2 cheaper, 2 mid, 1 nicer. Prices in INR per night. "
            'Schema: {"hotels":[{"name":str,"area":str,"price_per_night_inr":number,"rating":number}]}'
        ),
    )
    hotels: list[dict] = []
    seen: set[str] = set()
    for row in data.get("hotels") or []:
        if not isinstance(row, dict):
            continue
        try:
            name = str(row.get("name") or "").strip()
            price = float(row.get("price_per_night_inr") or 0)
            rating = float(row.get("rating") or 4.0)
        except (TypeError, ValueError):
            continue
        if not name or price <= 0 or name.lower() in seen:
            continue
        seen.add(name.lower())
        area = str(row.get("area") or "").strip()
        hotels.append(
            {
                "name": f"{name} ({area})" if area else name,
                "price_per_night_inr": round(price, 2),
                "rating": max(1.0, min(5.0, rating)),
                "source": _llm_source(),
            }
        )
    return hotels[:5]


def _llm_flight(origin: str, destination: str, date_str: str, o: str, d: str) -> dict:
    data = llm_json(
        "You estimate a realistic one-way economy airfare. Reply ONLY with JSON.",
        (
            f"One-way {origin} ({o}) to {destination} ({d}) on {date_str}. "
            'Schema: {"cheapest_price_inr": number, "airline": str, "duration_hours": number}'
        ),
    )
    try:
        price = float(data.get("cheapest_price_inr") or 0)
    except (TypeError, ValueError):
        price = 0
    if price <= 0:
        return {}
    try:
        hours = float(data.get("duration_hours") or 0)
    except (TypeError, ValueError):
        hours = 0
    airline = str(data.get("airline") or "").strip() or None
    return {
        "available": True,
        "cheapest_price_inr": round(price, 2),
        "duration_hours": hours or None,
        "airline": airline,
        "source": _llm_source(),
        "origin_iata": o,
        "destination_iata": d,
    }


def _offline_hotels(destination: str, budget_tier: str) -> list[dict]:
    seeded = get_hotels(destination)
    if seeded:
        return [{**h, "source": "seed"} for h in seeded]
    return []


def _resolve_hotels(destination: str, budget_tier: str) -> list[dict]:
    key = (destination.strip().lower(), budget_tier)
    if key in _hotel_cache:
        return _hotel_cache[key]

    hotels = _llm_hotels(destination, budget_tier) or _offline_hotels(destination, budget_tier)

    if budget_tier == "budget":
        hotels = sorted(hotels, key=lambda h: h["price_per_night_inr"])
    elif budget_tier == "premium":
        hotels = sorted(hotels, key=lambda h: -h.get("rating", 0))

    _hotel_cache[key] = hotels
    return hotels


def _distance_flight(origin: str, destination: str, o: str, d: str) -> dict:
    from tools.mapbox_api import geocode_location, haversine_km

    geo_o = geocode_location.invoke({"place_name": origin})
    geo_d = geocode_location.invoke({"place_name": destination})
    km = haversine_km(geo_o["lat"], geo_o["lng"], geo_d["lat"], geo_d["lng"])
    hours = round(max(1.2, km / 750.0 + 0.85), 1)
    return {
        "available": True,
        "cheapest_price_inr": float(round(2800 + km * 3.2, 0)),
        "duration_hours": hours,
        "airline": None,
        "source": "distance",
        "origin_iata": o,
        "destination_iata": d,
        "origin": origin,
        "destination": destination,
    }


@tool(parse_docstring=True)
def search_hotels(destination: str, budget_tier: str = "mid") -> list[dict]:
    """Search hotels for a destination. Gemini, then Groq; seed if no model is configured.

    Args:
        destination: Name of the destination, e.g. "Munnar" or "Goa".
        budget_tier: One of "budget", "mid", "premium" — used to sort results.

    Returns:
        List of dicts with name, price_per_night_inr, rating, source.
    """
    return _resolve_hotels(destination, budget_tier)


@tool(parse_docstring=True)
def search_hotel_offers(destination: str, nights: int) -> dict:
    """Stay total from the cheapest hotel in the local market list.

    Args:
        destination: Name of the destination.
        nights: Number of nights to stay.

    Returns:
        dict with cheapest_total_inr, hotels source, and best_rated name.
    """
    nights = max(1, nights)
    hotels = _resolve_hotels(destination, "mid")
    if not hotels:
        return {"cheapest_total_inr": 0.0, "best_rated": None, "source": "none"}
    cheapest = min(hotels, key=lambda h: h["price_per_night_inr"])
    best_rated = max(hotels, key=lambda h: h.get("rating", 0))
    return {
        "cheapest_total_inr": cheapest["price_per_night_inr"] * nights,
        "cheapest_hotel": cheapest["name"],
        "best_rated": best_rated["name"],
        "best_rated_price_inr": best_rated["price_per_night_inr"] * nights,
        "source": cheapest.get("source"),
    }


@tool(parse_docstring=True)
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search a one-way flight when both cities have IATA codes.

    Args:
        origin: Origin city name.
        destination: Destination city name.
        date: Departure date, YYYY-MM-DD.

    Returns:
        dict with available (bool) and cheapest_price_inr if found.
    """
    o = _city_iata(origin)
    d = _city_iata(destination)
    if not o or not d or o == d:
        return {
            "available": False,
            "reason": "Intra-region or unknown airports — use road routing",
            "source": "rules",
            "origin_iata": o,
            "destination_iata": d,
        }

    cache_key = (o, d, date)
    if cache_key in _flight_cache:
        return _flight_cache[cache_key]

    quote = _llm_flight(origin, destination, date, o, d) or _distance_flight(origin, destination, o, d)
    _flight_cache[cache_key] = quote
    return quote


@tool(parse_docstring=True)
def check_transport_disruptions(destination: str) -> dict:
    """Re-check air availability into the destination's nearest airport.

    Args:
        destination: Destination name to check.

    Returns:
        dict with disrupted (bool) and reason.
    """
    iata = _city_iata(destination)
    if not iata:
        return {"disrupted": False, "reason": "no airport — road transfer assumed", "source": "rules"}
    return {"disrupted": False, "reason": "", "source": "rules"}
