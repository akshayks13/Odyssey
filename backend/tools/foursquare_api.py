"""Places for a trip: the LLM lists the cities and their sights. The Foursquare helpers below are kept but not used:
its nearby search returns every kind of venue (schools, shops, clinics), so it is not a source of sights."""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config import FOURSQUARE_API_KEY
from llm import get_llm, llm_decide, llm_json, llm_provider_name
from tools.mapbox_api import geocode_location, haversine_km as mapbox_haversine
from tools import pmap

FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"
FSQ_PLACE_URL = "https://places-api.foursquare.com/places/{fsq_id}"
FSQ_PHOTOS_URL = "https://places-api.foursquare.com/places/{fsq_id}/photos"
FSQ_API_VERSION = "2025-06-17"

_TIMEOUT = 8.0
_DEFAULT_SCORES = {
    "nature": 0.5,
    "adventure": 0.4,
    "food": 0.4,
    "nightlife": 0.2,
    "relaxation": 0.5,
    "culture": 0.4,
    "shopping": 0.3,
}
_CATEGORIES = set(_DEFAULT_SCORES) | {"general"}
_attraction_cache: dict[tuple[str, str], list[dict]] = {}


def _fsq_headers() -> dict | None:
    if not FOURSQUARE_API_KEY:
        return None
    return {
        "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
        "Accept": "application/json",
        "X-Places-Api-Version": FSQ_API_VERSION,
    }


def _hours_from_details(data: dict) -> tuple[int, int] | None:
    hours = data.get("hours") or {}
    regular = hours.get("regular") or []
    opens: list[int] = []
    closes: list[int] = []
    for slot in regular:
        raw_open = str(slot.get("open") or slot.get("start") or "")
        raw_close = str(slot.get("close") or slot.get("end") or "")
        if len(raw_open) >= 2 and len(raw_close) >= 2:
            try:
                opens.append(int(raw_open[:2]))
                closes.append(int(raw_close[:2]) or 23)
            except ValueError:
                continue
    if not opens:
        return None
    return min(opens), max(closes)


def _coords_from_place(place: dict) -> tuple[float | None, float | None]:
    lat, lng = place.get("latitude"), place.get("longitude")
    if lat is None or lng is None:
        geo = (place.get("geocodes") or {}).get("main") or {}
        lat, lng = geo.get("latitude"), geo.get("longitude")
    if lat is None or lng is None:
        loc = place.get("location") or {}
        lat, lng = loc.get("latitude") or loc.get("lat"), loc.get("longitude") or loc.get("lng")
    if lat is None or lng is None:
        return None, None
    return float(lat), float(lng)


def _place_to_activity(place: dict, destination: str, category: str | None) -> dict:
    cats = place.get("categories") or []
    cat_name = category or ((cats[0].get("name") if cats else None) or "general")
    cat_l = str(cat_name).lower()
    mapped = "general"
    for key in _DEFAULT_SCORES:
        if key in cat_l:
            mapped = key
            break
    if "relax" in cat_l:
        mapped = "relaxation"
    lat, lng = _coords_from_place(place)
    hours = _hours_from_details(place)
    return {
        "id": place.get("fsq_place_id") or place.get("fsq_id") or place.get("name", "unknown"),
        "name": place.get("name", "Unknown"),
        "destination": destination,
        "category": mapped,
        "duration_minutes": 120,
        "cost_inr": 0.0,
        "rating": float(place.get("rating") or 4.0),
        "opening_hour": hours[0] if hours else 9,
        "closing_hour": hours[1] if hours else 18,
        "preference_score": 0.6,
        "coordinates": {"lat": lat, "lng": lng} if lat is not None else None,
        "source": "foursquare",
    }


def _enrich_hours(activity: dict) -> dict:
    headers = _fsq_headers()
    fsq_id = activity.get("id")
    if not headers or not fsq_id or activity.get("source") != "foursquare":
        return activity
    try:
        resp = httpx.get(
            FSQ_PLACE_URL.format(fsq_id=fsq_id),
            headers=headers,
            params={"fields": "hours,rating,geocodes,name,categories"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        details = resp.json()
        hours = _hours_from_details(details)
        if hours:
            activity["opening_hour"], activity["closing_hour"] = hours
        if details.get("rating"):
            activity["rating"] = float(details["rating"])
        lat, lng = _coords_from_place(details)
        if lat is not None:
            activity["coordinates"] = {"lat": lat, "lng": lng}
    except Exception:
        pass
    return activity


def _fsq_search(destination: str, category: str | None, lat: float | None, lng: float | None) -> list[dict]:
    headers = _fsq_headers()
    if not headers:
        return []
    params: dict = {"limit": 15}
    if lat is not None and lng is not None:
        params["ll"] = f"{lat},{lng}"
        params["radius"] = 20000
    else:
        params["near"] = destination
    if category:
        params["query"] = category
    try:
        resp = httpx.get(FSQ_SEARCH_URL, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("results") or []
    except Exception:
        return []


def _to_float(value) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _activity_from_llm(row: dict, destination: str, fallback_lat: float | None, fallback_lng: float | None) -> dict | None:
    name = str(row.get("name") or "").strip()
    if not name:
        return None
    category = str(row.get("category") or "general").lower().strip()
    if category not in _CATEGORIES:
        category = "general"
    try:
        lat = float(row["lat"]) if row.get("lat") is not None else None
        lng = float(row["lng"]) if row.get("lng") is not None else None
    except (TypeError, ValueError):
        lat, lng = None, None
    if lat is None or lng is None:
        geo = geocode_location.invoke({"place_name": f"{name}, {destination}"})
        lat, lng = geo.get("lat"), geo.get("lng")
    if lat is None:
        lat, lng = fallback_lat, fallback_lng
    try:
        duration = int(row.get("duration_minutes") or 120)
    except (TypeError, ValueError):
        duration = 120
    try:
        opening = int(row.get("opening_hour") or 9)
        closing = int(row.get("closing_hour") or 18)
    except (TypeError, ValueError):
        opening, closing = 9, 18
    try:
        rating = float(row.get("rating") or 4.2)
    except (TypeError, ValueError):
        rating = 4.2
    return {
        "id": f"llm-{name.lower().replace(' ', '-')[:48]}",
        "name": name,
        "destination": destination,
        "category": category,
        "duration_minutes": max(45, min(duration, 360)),
        "cost_inr": _to_float(row.get("cost_inr")),
        "rating": max(1.0, min(5.0, rating)),
        "opening_hour": max(6, min(opening, 22)),
        "closing_hour": max(opening + 1, min(closing, 23)),
        "preference_score": 0.7,
        "coordinates": {"lat": lat, "lng": lng} if lat is not None else None,
        "source": (llm_provider_name() or "llm").split(":")[0],
    }


def _llm_attractions(destination: str, category: str | None, lat: float | None, lng: float | None) -> list[dict]:
    llm = get_llm()
    if llm is None:
        return []
    focus = f" Prefer the '{category}' theme." if category else ""
    decision = llm_decide(
        llm,
        tools=[],
        system=(
            "You list real visitor attractions for a travel itinerary. "
            "Reply ONLY with JSON: {\"attractions\":[{\"name\":str,\"category\":"
            "\"nature|adventure|culture|relaxation|food|shopping|nightlife\","
            "\"duration_minutes\":int,\"rating\":number,\"opening_hour\":int,"
            "\"closing_hour\":int,\"cost_inr\":int,\"lat\":number,\"lng\":number}]}. "
            "cost_inr is the per-person entry fee in INR for an Indian resident (0 if free)."
        ),
        user=(
            f"City: {destination}.{focus} "
            "Return 6-10 well-known parks, lakes, gardens, viewpoints, museums, temples "
            "or historic sites in that city. Coordinates must be in the city. "
            "Do not list hotel chains, clinics, or fast-food."
        ),
        max_rounds=1,
    )
    out: list[dict] = []
    for row in decision.get("attractions") or []:
        if not isinstance(row, dict):
            continue
        act = _activity_from_llm(row, destination, lat, lng)
        if act:
            out.append(act)
    return out


_destination_cache: dict[str, list[dict]] = {}


def _llm_destinations(region: str) -> list[dict]:
    """Ask the model which places in a region a traveller would base themselves in.

    Coordinates are taken from Mapbox when available; a model's lat/lng for a small town can be off.
    """
    data = llm_json(
        "You are a travel-destination expert. You list real places within a region that a traveller "
        "would stay in or day-trip to. Reply ONLY with JSON.",
        (
            f"Region: {region}. List 6-9 distinct real places (towns, cities, hill stations, beaches, "
            "islands, districts) in or next to this region, best-known first. If the region is itself a "
            "single city or town, list it first and then up to 4 nearby places worth an overnight or a "
            "day trip. Coordinates must be accurate. activity_scores are 0-1 for how strongly the place "
            "suits each theme. "
            'Schema: {"places":[{"name":str,"lat":number,"lng":number,"description":str,'
            '"tags":[str],"activity_scores":{"nature":n,"adventure":n,"food":n,"nightlife":n,'
            '"relaxation":n,"culture":n,"shopping":n}}]}'
        ),
    )
    rows = [r for r in (data.get("places") or []) if isinstance(r, dict) and str(r.get("name") or "").strip()]
    # Geocode with the whole region ("Palolem, Goa, India"): "Palolem, India" alone finds a different village.
    geos = pmap(lambda r: geocode_location.invoke({"place_name": f"{str(r['name']).strip()}, {region}"}), rows[:10], workers=5)
    out: list[dict] = []
    seen: set[str] = set()
    for row, geo in zip(rows, geos):
        name = str(row["name"]).strip()
        if name.lower() in seen:
            continue
        try:
            lat, lng = float(row["lat"]), float(row["lng"])
        except (KeyError, TypeError, ValueError):
            lat = lng = None
        if geo and geo.get("source") in {"mapbox", "osm"}:
            # Two independent answers: if they are far apart one is wrong. The model's own coordinates
            # are kept then, because a geocoder picks a same-named village elsewhere far more often.
            agree = lat is None or mapbox_haversine(lat, lng, geo["lat"], geo["lng"]) <= 60
            if agree:
                lat, lng = geo["lat"], geo["lng"]
        if lat is None or lng is None:
            continue
        raw_scores = row.get("activity_scores") if isinstance(row.get("activity_scores"), dict) else {}
        scores = {k: max(0.0, min(1.0, _to_float(raw_scores.get(k, v)))) for k, v in _DEFAULT_SCORES.items()}
        seen.add(name.lower())
        out.append(
            {
                "name": name,
                "lat": lat,
                "lng": lng,
                "description": str(row.get("description") or name)[:200],
                "tags": [str(t) for t in (row.get("tags") or [])][:6],
                "activity_scores": scores,
                "source": (llm_provider_name() or "llm").split(":")[0],
            }
        )
    return out[:9]


@tool(parse_docstring=True)
def search_destinations(region: str) -> list[dict]:
    """Find candidate places to base a trip in a region.

    Asks the language model for real places in the region (any country). Empty if it cannot answer.

    Args:
        region: Region or city name, e.g. "Kerala, India" or "Tuscany, Italy".

    Returns:
        List of dicts with name, lat, lng, description, tags, activity_scores.
    """
    cache_key = region.strip().lower()
    if cache_key in _destination_cache:
        return _destination_cache[cache_key]

    rows = _llm_destinations(region)
    if rows:  # an empty answer is never remembered, so the next call asks again
        _destination_cache[cache_key] = rows
    return rows


@tool(parse_docstring=True)
def search_attractions(
    destination: str,
    category: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
) -> list[dict]:
    """Search visitor attractions in a destination.

    Args:
        destination: Name of the destination, e.g. "Munnar".
        category: Optional category filter, e.g. "nature" or "adventure".
        lat: Optional latitude, used when a sight comes back without coordinates.
        lng: Optional longitude, used when a sight comes back without coordinates.

    Returns:
        List of activity dicts with id, name, category, duration_minutes,
        cost_inr, rating, opening_hour, closing_hour, preference_score.
    """
    cache_key = (destination.strip().lower(), (category or "").strip().lower())
    if cache_key in _attraction_cache:
        return _attraction_cache[cache_key]

    # The LLM is the source of sights. Foursquare's nearby search returns every kind of venue
    # (schools, shops, clinics), so its helpers are kept in this file but not used here.
    out = _llm_attractions(destination, category, lat, lng)
    if out:  # don't remember a failed lookup
        _attraction_cache[cache_key] = out
    return out


__all__ = ["search_destinations", "search_attractions"]
