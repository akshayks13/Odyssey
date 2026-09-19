"""Hotels and flights for any destination.

When Gemini is available the market tools ask it for realistic local hotels
and airfares. Otherwise we use the Kerala seed (known towns) or a
deterministic heuristic so any city still gets a usable list.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

from langchain_core.tools import tool

from tools.seed_data import get_hotels

_CITY_IATA = {
    "kochi": "COK",
    "cochin": "COK",
    "ernakulam": "COK",
    "trivandrum": "TRV",
    "thiruvananthapuram": "TRV",
    "kovalam": "TRV",
    "varkala": "TRV",
    "calicut": "CCJ",
    "kozhikode": "CCJ",
    "wayanad": "CCJ",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "chennai": "MAA",
    "hyderabad": "HYD",
    "goa": "GOI",
}

_hotel_cache: dict[tuple[str, str], list[dict]] = {}
_flight_cache: dict[tuple[str, str, str], dict] = {}


def _city_iata(name: str) -> str | None:
    return _CITY_IATA.get(name.strip().lower())


def _digest(text: str) -> int:
    return int(hashlib.md5(text.lower().encode("utf-8")).hexdigest()[:8], 16)


def _heuristic_hotels(destination: str, budget_tier: str) -> list[dict]:
    base = {"budget": 1400, "mid": 2800, "premium": 6200}.get(budget_tier, 2800)
    jitter = _digest(destination) % 800
    stems = ("Central Inn", "Garden Stay", "Heritage Lodge", "Lakeside Residency", "Hillview Hotel")
    hotels = []
    for i, stem in enumerate(stems):
        hotels.append(
            {
                "name": f"{destination} {stem}",
                "price_per_night_inr": float(base + jitter + i * 350),
                "rating": round(3.6 + (i % 4) * 0.3, 1),
                "source": "generated",
            }
        )
    return hotels


def _heuristic_flight(origin: str, destination: str, o: str, d: str) -> dict:
    km_proxy = 400 + (_digest(f"{o}-{d}") % 1600)
    return {
        "available": True,
        "cheapest_price_inr": float(3500 + km_proxy * 4),
        "source": "generated",
        "origin_iata": o,
        "destination_iata": d,
        "origin": origin,
        "destination": destination,
    }


def _llm_json(system: str, user: str) -> dict:
    """Ask Gemini for a JSON object. Fail fast onto heuristic/seed on 429/timeout."""
    from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_DISABLED
    from llm import parse_json_blob

    if LLM_DISABLED or not GEMINI_API_KEY:
        return {}
    try:
        import httpx

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
                "generationConfig": {"temperature": 0.3},
            },
            timeout=12.0,
        )
        if resp.status_code >= 400:
            return {}
        parts = (((resp.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        text = "".join(p.get("text") or "" for p in parts)
        return parse_json_blob(text)
    except Exception:
        return {}


def _gemini_hotels(destination: str, budget_tier: str) -> list[dict]:
    data = _llm_json(
        "You generate plausible but realistic hotel inventory for a travel planner. "
        "Reply ONLY with JSON.",
        (
            f"List 5 real-feeling hotels in {destination} for a {budget_tier}-tier traveller. "
            "Prices in INR per night, typical for that city (not luxury outliers). "
            'Schema: {"hotels":[{"name":str,"price_per_night_inr":number,"rating":number}]}'
        ),
    )
    hotels: list[dict] = []
    for row in data.get("hotels") or []:
        try:
            name = str(row.get("name") or "").strip()
            price = float(row.get("price_per_night_inr") or 0)
            rating = float(row.get("rating") or 4.0)
        except (TypeError, ValueError):
            continue
        if not name or price <= 0:
            continue
        hotels.append(
            {
                "name": name,
                "price_per_night_inr": round(price, 2),
                "rating": max(1.0, min(5.0, rating)),
                "source": "gemini",
            }
        )
    return hotels[:5]


def _gemini_flight(origin: str, destination: str, date_str: str, o: str, d: str) -> dict:
    data = _llm_json(
        "You estimate a realistic one-way economy airfare. Reply ONLY with JSON.",
        (
            f"One-way {origin} ({o}) to {destination} ({d}) on {date_str}. "
            'Schema: {"cheapest_price_inr": number, "airline": str}'
        ),
    )
    try:
        price = float(data.get("cheapest_price_inr") or 0)
    except (TypeError, ValueError):
        price = 0
    if price <= 0:
        return {}
    return {
        "available": True,
        "cheapest_price_inr": round(price, 2),
        "airline": data.get("airline"),
        "source": "gemini",
        "origin_iata": o,
        "destination_iata": d,
    }


def _resolve_hotels(destination: str, budget_tier: str) -> list[dict]:
    key = (destination.strip().lower(), budget_tier)
    if key in _hotel_cache:
        return _hotel_cache[key]

    hotels = _gemini_hotels(destination, budget_tier)
    if not hotels:
        seeded = get_hotels(destination)
        hotels = [{**h, "source": "seed"} for h in seeded] if seeded else _heuristic_hotels(destination, budget_tier)

    if budget_tier == "budget":
        hotels = sorted(hotels, key=lambda h: h["price_per_night_inr"])
    elif budget_tier == "premium":
        hotels = sorted(hotels, key=lambda h: -h.get("rating", 0))

    _hotel_cache[key] = hotels
    return hotels


@tool(parse_docstring=True)
def search_hotels(destination: str, budget_tier: str = "mid") -> list[dict]:
    """Search hotels for a destination. Uses Gemini market estimates, then seed/heuristic.

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
    """Search long-haul/intercity air (not intra-Kerala road hops like Kochi–Munnar).

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

    quote = _gemini_flight(origin, destination, date, o, d) or _heuristic_flight(origin, destination, o, d)
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
    probe_date = (date.today() + timedelta(days=21)).isoformat()
    offers = search_flights.invoke({"origin": "Delhi", "destination": destination, "date": probe_date})
    if not offers.get("available"):
        return {"disrupted": False, "reason": offers.get("reason", ""), "source": offers.get("source")}
    return {"disrupted": False, "reason": "", "source": offers.get("source")}
