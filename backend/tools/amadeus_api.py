"""Amadeus Self-Service tools (test sandbox).

Plan endpoints:
  - Hotel List by city / by geocode
  - Hotel Offers (live nightly price)
  - Flight Offers (long-haul only)

City codes are resolved properly (IATA map + Amadeus city search).
We never slice `destination[:3]` — that turned "Munnar" into invalid "MUN"
and made every live call fail, so the tool silently used seed hotels.

Seed data is the last resort when credentials are missing, OAuth fails, or
the sandbox has no inventory for that hill town (common for India).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import httpx
from langchain_core.tools import tool

from config import AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET
from tools.seed_data import get_destination_seed, get_hotels

AMADEUS_HOST = "https://test.api.amadeus.com"
AUTH_URL = f"{AMADEUS_HOST}/v1/security/oauth2/token"
CITIES_URL = f"{AMADEUS_HOST}/v1/reference-data/locations/cities"
HOTELS_BY_CITY_URL = f"{AMADEUS_HOST}/v1/reference-data/locations/hotels/by-city"
HOTELS_BY_GEO_URL = f"{AMADEUS_HOST}/v1/reference-data/locations/hotels/by-geocode"
HOTEL_OFFERS_URL = f"{AMADEUS_HOST}/v3/shopping/hotel-offers"
FLIGHT_OFFERS_URL = f"{AMADEUS_HOST}/v2/shopping/flight-offers"

_TIMEOUT = 12.0
_token_cache: dict = {"token": None, "expires_at": 0}

# Real IATA city/airport codes — NOT the first three letters of the English name.
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

_FX_TO_INR = {"INR": 1.0, "EUR": 90.0, "USD": 83.0, "GBP": 105.0}


def _to_inr(amount: float, currency: str) -> float:
    return round(float(amount) * _FX_TO_INR.get(currency.upper(), 83.0), 2)


def _get_access_token() -> str | None:
    if not (AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET):
        return None
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]
    try:
        resp = httpx.post(
            AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": AMADEUS_CLIENT_ID,
                "client_secret": AMADEUS_CLIENT_SECRET,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + int(data.get("expires_in", 1800)) - 60
        return _token_cache["token"]
    except Exception:
        return None


def _auth_headers() -> dict | None:
    token = _get_access_token()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _city_iata(name: str) -> str | None:
    key = name.strip().lower()
    if key in _CITY_IATA:
        return _CITY_IATA[key]
    headers = _auth_headers()
    if not headers:
        return None
    try:
        resp = httpx.get(
            CITIES_URL,
            headers=headers,
            params={"keyword": name, "countryCode": "IN", "max": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []
        if rows:
            return rows[0].get("iataCode")
    except Exception:
        return None
    return None


def _coords(destination: str) -> tuple[float, float] | None:
    seed = get_destination_seed(destination)
    if seed:
        return seed["coordinates"]["lat"], seed["coordinates"]["lng"]
    return None


def _list_hotels_live(destination: str) -> tuple[list[dict], str | None]:
    """Return (hotel stubs with hotelId), error_message."""
    headers = _auth_headers()
    if not headers:
        return [], "missing AMADEUS_CLIENT_ID/SECRET"

    coords = _coords(destination)
    if coords:
        try:
            resp = httpx.get(
                HOTELS_BY_GEO_URL,
                headers=headers,
                params={"latitude": coords[0], "longitude": coords[1], "radius": 30, "radiusUnit": "KM"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            rows = resp.json().get("data") or []
            if rows:
                return [_hotel_stub(h) for h in rows[:8]], None
        except Exception as exc:
            geo_err = str(exc)
        else:
            geo_err = "empty geocode hotel list"
    else:
        geo_err = "no coordinates"

    city = _city_iata(destination)
    if not city:
        return [], f"no IATA city code ({geo_err})"
    try:
        resp = httpx.get(
            HOTELS_BY_CITY_URL,
            headers=headers,
            params={"cityCode": city},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []
        if rows:
            return [_hotel_stub(h) for h in rows[:8]], None
        return [], f"empty hotel list for cityCode={city}"
    except Exception as exc:
        return [], f"by-city failed: {exc}"


def _hotel_stub(h: dict) -> dict:
    return {
        "hotel_id": h.get("hotelId") or h.get("id"),
        "name": (h.get("name") or h.get("hotel", {}).get("name") or "Hotel"),
        "rating": float(h.get("rating") or 4.0),
        "source": "amadeus",
    }


def _price_hotels(hotels: list[dict], nights: int) -> list[dict]:
    """Call Hotel Offers for live nightly rates; keep unpriced hotels if offers fail."""
    headers = _auth_headers()
    ids = [h["hotel_id"] for h in hotels if h.get("hotel_id")]
    if not headers or not ids:
        return hotels

    check_in = date.today() + timedelta(days=21)
    check_out = check_in + timedelta(days=max(1, nights))
    priced: dict[str, float] = {}
    try:
        resp = httpx.get(
            HOTEL_OFFERS_URL,
            headers=headers,
            params={
                "hotelIds": ",".join(ids[:5]),
                "adults": 1,
                "roomQuantity": 1,
                "checkInDate": check_in.isoformat(),
                "checkOutDate": check_out.isoformat(),
                "currency": "INR",
                "bestRateOnly": True,
            },
            timeout=_TIMEOUT,
        )
        if resp.status_code >= 400:
            # Sandbox often rejects INR — retry in EUR and convert.
            resp = httpx.get(
                HOTEL_OFFERS_URL,
                headers=headers,
                params={
                    "hotelIds": ",".join(ids[:5]),
                    "adults": 1,
                    "checkInDate": check_in.isoformat(),
                    "checkOutDate": check_out.isoformat(),
                    "currency": "EUR",
                    "bestRateOnly": True,
                },
                timeout=_TIMEOUT,
            )
        resp.raise_for_status()
        for row in resp.json().get("data") or []:
            hid = (row.get("hotel") or {}).get("hotelId")
            offers = row.get("offers") or []
            if not hid or not offers:
                continue
            price = offers[0].get("price") or {}
            total = float(price.get("total") or 0)
            currency = price.get("currency") or "EUR"
            stay_inr = _to_inr(total, currency)
            priced[hid] = round(stay_inr / max(1, nights), 2)
    except Exception:
        priced = {}

    out = []
    for h in hotels:
        nightly = priced.get(h.get("hotel_id"), 3000.0 if not priced else None)
        if nightly is None:
            continue
        out.append({**h, "price_per_night_inr": nightly, "source": "amadeus"})
    return out or [{**h, "price_per_night_inr": 3000.0, "source": "amadeus_unpriced"} for h in hotels]


def _seed_hotels(destination: str, reason: str) -> list[dict]:
    hotels = get_hotels(destination)
    return [
        {**h, "source": "seed", "amadeus_fallback": reason} for h in hotels
    ] or [
        {
            "name": f"{destination} Standard Stay",
            "price_per_night_inr": 2000.0,
            "rating": 3.8,
            "source": "seed_default",
            "amadeus_fallback": reason,
        }
    ]


@tool(parse_docstring=True)
def search_hotels(destination: str, budget_tier: str = "mid") -> list[dict]:
    """Search hotels via Amadeus Hotel List (geocode, then city code) plus Hotel Offers pricing.

    Args:
        destination: Name of the destination, e.g. "Munnar" or "Kochi".
        budget_tier: One of "budget", "mid", "premium" — used only to sort results.

    Returns:
        List of dicts with name, price_per_night_inr, rating, source.
    """
    live, err = _list_hotels_live(destination)
    if live:
        priced = _price_hotels(live, nights=1)
        if budget_tier == "budget":
            priced.sort(key=lambda h: h["price_per_night_inr"])
        elif budget_tier == "premium":
            priced.sort(key=lambda h: -h.get("rating", 0))
        return priced
    return _seed_hotels(destination, err or "amadeus returned no hotels")


@tool(parse_docstring=True)
def search_hotel_offers(destination: str, nights: int) -> dict:
    """Live-priced stay total from Amadeus Hotel Offers.

    Args:
        destination: Name of the destination.
        nights: Number of nights to stay.

    Returns:
        dict with cheapest_total_inr, hotels source, and best_rated name.
    """
    live, err = _list_hotels_live(destination)
    hotels = _price_hotels(live, nights=max(1, nights)) if live else _seed_hotels(destination, err or "no live hotels")
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
        "amadeus_fallback": cheapest.get("amadeus_fallback"),
    }


@tool(parse_docstring=True)
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search Amadeus Flight Offers for long-haul/intercity air (not Kochi–Munnar road hops).

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

    headers = _auth_headers()
    if not headers:
        return {"available": False, "reason": "missing Amadeus credentials", "source": "seed"}

    try:
        resp = httpx.get(
            FLIGHT_OFFERS_URL,
            headers=headers,
            params={
                "originLocationCode": o,
                "destinationLocationCode": d,
                "departureDate": date,
                "adults": 1,
                "max": 3,
                "currencyCode": "INR",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if data:
            price = data[0].get("price") or {}
            total = float(price.get("grandTotal") or price.get("total") or 0)
            currency = price.get("currency") or "INR"
            return {
                "available": True,
                "cheapest_price_inr": _to_inr(total, currency),
                "source": "amadeus",
                "origin_iata": o,
                "destination_iata": d,
            }
        return {"available": False, "reason": "no offers in sandbox", "source": "amadeus", "origin_iata": o, "destination_iata": d}
    except Exception as exc:
        return {"available": False, "reason": str(exc), "source": "amadeus_error", "origin_iata": o, "destination_iata": d}


@tool(parse_docstring=True)
def check_transport_disruptions(destination: str) -> dict:
    """Re-check Amadeus flight availability into the destination's nearest airport.

    Args:
        destination: Destination name to check.

    Returns:
        dict with disrupted (bool) and reason.
    """
    iata = _city_iata(destination)
    if not iata:
        return {"disrupted": False, "reason": "no airport — road transfer assumed", "source": "rules"}
    # Probe a common long-haul inbound (DEL → dest) a few weeks out.
    probe_date = (date.today() + timedelta(days=21)).isoformat()
    offers = search_flights.invoke({"origin": "Delhi", "destination": destination, "date": probe_date})
    if offers.get("source") == "amadeus_error":
        return {"disrupted": True, "reason": offers.get("reason", "Amadeus flight lookup failed"), "source": "amadeus"}
    return {"disrupted": False, "reason": offers.get("reason", ""), "source": offers.get("source")}
