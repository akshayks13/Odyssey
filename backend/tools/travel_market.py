"""Hotel and public-transport quotes from the LLM (each Gemini key in turn, then Groq). No model answer
means no quote — never an invented hotel or fare."""
from __future__ import annotations

from langchain_core.tools import tool

from llm import llm_json, llm_provider_name

_hotel_cache: dict[tuple[str, str], list[dict]] = {}
_transport_cache: dict[tuple[str, str, str], dict] = {}


def _llm_source() -> str:
    return (llm_provider_name() or "llm").split(":")[0]


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


def _resolve_hotels(destination: str, budget_tier: str) -> list[dict]:
    key = (destination.strip().lower(), budget_tier)
    if key in _hotel_cache:
        return _hotel_cache[key]

    hotels = _llm_hotels(destination, budget_tier)
    if budget_tier == "budget":
        hotels = sorted(hotels, key=lambda h: h["price_per_night_inr"])
    elif budget_tier == "premium":
        hotels = sorted(hotels, key=lambda h: -h.get("rating", 0))

    if hotels:  # an empty result is a failed lookup; don't remember it
        _hotel_cache[key] = hotels
    return hotels


@tool(parse_docstring=True)
def search_hotels(destination: str, budget_tier: str = "mid") -> list[dict]:
    """Search hotels for a destination (LLM market estimate).

    Args:
        destination: Name of the destination, e.g. "Jaipur" or "Goa".
        budget_tier: One of "budget", "mid", "premium" — used to sort results.

    Returns:
        List of dicts with name, price_per_night_inr, rating, source.
    """
    return _resolve_hotels(destination, budget_tier)


@tool(parse_docstring=True)
def search_public_transport(origin: str, destination: str, mode: str, date: str = "") -> dict:
    """Estimate a one-way flight or train journey between two places.

    The model knows which places have airports or stations, so a town without one is quoted as
    "ride to the nearest airport, fly, ride on" with the transfer time included.

    Args:
        origin: Origin place.
        destination: Destination place.
        mode: "air" or "rail".
        date: Optional travel date, YYYY-MM-DD.

    Returns:
        dict with available, duration_hours (door to door), cost_inr (per person, whole journey),
        operator, summary. available is False when no realistic service exists.
    """
    mode = "rail" if (mode or "").lower() == "rail" else "air"
    key = (origin.strip().lower(), destination.strip().lower(), mode)
    if key in _transport_cache:
        return _transport_cache[key]

    kind = "flight" if mode == "air" else "train"
    data = llm_json(
        "You estimate realistic one-way travel quotes in India and abroad. Reply ONLY with JSON.",
        (
            f"One-way {kind} from {origin} to {destination}" + (f" around {date}" if date else "") + ", "
            "one adult, standard economy fare. If either place has no airport/station, include the ride to "
            "the nearest one in duration_hours and price_inr and say so in note. If no realistic service "
            'exists, set available to false. Schema: {"available": bool, "operator": str, '
            '"duration_hours": number (door to door), "price_inr": number (per person), "note": str}'
        ),
    )
    try:
        hours, price = float(data.get("duration_hours") or 0), float(data.get("price_inr") or 0)
    except (TypeError, ValueError):
        hours = price = 0.0
    if not data:
        return {"available": False, "mode": mode, "reason": "no model available to quote this"}
    if not data.get("available") or hours <= 0 or price <= 0:
        quote = {"available": False, "mode": mode, "reason": str(data.get("note") or "no realistic service")}
    else:
        operator = str(data.get("operator") or "").strip()
        note = str(data.get("note") or "").strip()
        quote = {
            "available": True,
            "mode": mode,
            "duration_hours": round(hours, 2),
            "cost_inr": round(price, 2),
            "operator": operator,
            "summary": f"{operator + ' ' if operator else ''}{kind} {origin}→{destination} · {hours:.1f}h · ₹{int(price):,}"
            + (f" ({note})" if note else ""),
            "source": _llm_source(),
        }
        _transport_cache[key] = quote  # only a real quote is remembered; "unavailable" is asked again next time
    return quote


@tool(parse_docstring=True)
def check_transport_disruptions(destination: str, start_date: str = "") -> dict:
    """Check whether a storm at a destination is likely to disrupt travel there.

    Args:
        destination: Destination name to check.
        start_date: Travel date, YYYY-MM-DD. Defaults to today.

    Returns:
        dict with disrupted (bool), reason, and source.
    """
    from tools.weather import trip_weather

    weather = trip_weather(destination, start_date, 1)
    stormy = [r for r in weather["days"] if any(w in r["condition"].lower() for w in ("thunder", "violent", "heavy snow"))]
    if stormy:
        return {"disrupted": True, "reason": f"{stormy[0]['condition']} expected at {destination}", "source": weather["source"]}
    return {"disrupted": False, "reason": "", "source": weather["source"]}
