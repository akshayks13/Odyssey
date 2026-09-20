"""Foursquare Places: nearby search, hours, and photos."""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config import FOURSQUARE_API_KEY
from llm import get_llm, llm_decide, llm_provider_name
from tools.mapbox_api import geocode_location, get_static_map_url
from tools.seed_data import all_destinations, get_activities

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
        "cost_inr": float(row.get("cost_inr") or 0),
        "rating": max(1.0, min(5.0, rating)),
        "opening_hour": max(6, min(opening, 22)),
        "closing_hour": max(opening + 1, min(closing, 23)),
        "preference_score": 0.7,
        "coordinates": {"lat": lat, "lng": lng} if lat is not None else None,
        "source": (llm_provider_name() or "llm").split(":")[0],
    }


def _llm_attractions(
    destination: str,
    category: str | None,
    foursquare_names: list[str],
    lat: float | None,
    lng: float | None,
) -> list[dict]:
    llm = get_llm()
    if llm is None:
        return []
    hint = ", ".join(foursquare_names[:12]) if foursquare_names else "(none)"
    focus = f" Prefer the '{category}' theme." if category else ""
    decision = llm_decide(
        llm,
        tools=[],
        system=(
            "You list real visitor attractions for a travel itinerary. "
            "Reply ONLY with JSON: {\"attractions\":[{\"name\":str,\"category\":"
            "\"nature|adventure|culture|relaxation|food|shopping|nightlife\","
            "\"duration_minutes\":int,\"rating\":number,\"opening_hour\":int,"
            "\"closing_hour\":int,\"lat\":number,\"lng\":number}]}."
        ),
        user=(
            f"City: {destination}.{focus} "
            f"Foursquare nearby names (often restaurants/shops — keep only if they are "
            f"actual sights a traveller would visit): {hint}. "
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


def _seed_attractions(destination: str, category: str | None) -> list[dict]:
    """Bundled activities for this city, if the dataset has any."""
    dest = next((d for d in all_destinations() if d["name"].lower() == destination.lower()), None)
    if dest is None:
        return []
    activities = get_activities(dest["name"])
    if category:
        activities = [a for a in activities if a["category"] == category] or activities
    coords = dest["coordinates"]
    out = []
    for a in activities:
        row = {**a, "destination": destination, "source": "seed"}
        if coords and not row.get("coordinates"):
            row["coordinates"] = coords
        out.append(row)
    return out


@tool(parse_docstring=True)
def search_destinations(region: str) -> list[dict]:
    """Find candidate cities in a region.

    Uses bundled cities when the region matches that dataset, otherwise
    Mapbox geocoding. Attractions are fetched separately.

    Args:
        region: Region or city name, e.g. "Kerala, India".

    Returns:
        List of dicts with name, lat, lng, description, tags, activity_scores.
    """
    key = region.lower().split(",")[0].strip()
    seed_hits = [
        {
            "name": d["name"],
            "lat": d["coordinates"]["lat"],
            "lng": d["coordinates"]["lng"],
            "description": d["description"],
            "tags": d["tags"],
            "activity_scores": d["activity_scores"],
            "source": "seed",
        }
        for d in all_destinations()
        if key in d.get("region", "").lower() or key in d["name"].lower()
    ]
    if seed_hits:
        return seed_hits

    geo = geocode_location.invoke({"place_name": region})
    return [
        {
            "name": geo.get("name") or region.split(",")[0].strip(),
            "lat": geo["lat"],
            "lng": geo["lng"],
            "description": f"Trip base for {region}",
            "tags": [],
            "activity_scores": dict(_DEFAULT_SCORES),
            "source": geo.get("source", "mapbox"),
        }
    ]


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
        lat: Optional latitude to search around.
        lng: Optional longitude to search around.

    Returns:
        List of activity dicts with id, name, category, duration_minutes,
        cost_inr, rating, opening_hour, closing_hour, preference_score.
    """
    cache_key = (destination.strip().lower(), (category or "").strip().lower())
    if cache_key in _attraction_cache:
        return _attraction_cache[cache_key]

    fsq_places = _fsq_search(destination, category, lat, lng)
    fsq_acts = [_place_to_activity(p, destination, category) for p in fsq_places if p.get("name")]
    for act in fsq_acts[:8]:
        _enrich_hours(act)

    llm_acts = _llm_attractions(
        destination,
        category,
        [a["name"] for a in fsq_acts],
        lat,
        lng,
    )
    out = llm_acts or fsq_acts or _seed_attractions(destination, category)
    _attraction_cache[cache_key] = out
    return out


@tool(parse_docstring=True)
def get_opening_hours(activity_id: str, destination: str) -> dict:
    """Get opening/closing hours via Foursquare Places Details, else seed data.

    Args:
        activity_id: The activity's id (Foursquare fsq_id or seed id).
        destination: Destination the activity belongs to (used for seed lookup).

    Returns:
        dict with opening_hour and closing_hour (24h clock).
    """
    headers = _fsq_headers()
    if headers:
        try:
            resp = httpx.get(
                FSQ_PLACE_URL.format(fsq_id=activity_id),
                headers=headers,
                params={"fields": "hours,name"},
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                hours = _hours_from_details(resp.json())
                if hours:
                    return {
                        "opening_hour": hours[0],
                        "closing_hour": hours[1],
                        "source": "foursquare",
                    }
        except Exception:
            pass

    for act in get_activities(destination):
        if act["id"] == activity_id:
            return {
                "opening_hour": act["opening_hour"],
                "closing_hour": act["closing_hour"],
                "source": "seed",
            }
    return {"opening_hour": 9, "closing_hour": 18, "source": "default"}


@tool(parse_docstring=True)
def get_place_photos(place_name: str, lat: float | None = None, lng: float | None = None, fsq_id: str | None = None) -> dict:
    """Photos for a place from Foursquare, else a Mapbox Static image.

    Args:
        place_name: Human-readable place name (used to geocode if lat/lng omitted).
        lat: Optional latitude.
        lng: Optional longitude.
        fsq_id: Optional Foursquare place id.

    Returns:
        dict with photos (list of urls) and source.
    """
    headers = _fsq_headers()
    if headers and fsq_id:
        try:
            resp = httpx.get(FSQ_PHOTOS_URL.format(fsq_id=fsq_id), headers=headers, params={"limit": 3}, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            urls = []
            photos = payload if isinstance(payload, list) else payload.get("photos") or []
            for photo in photos:
                prefix, suffix = photo.get("prefix"), photo.get("suffix")
                if prefix and suffix:
                    urls.append(f"{prefix}original{suffix}")
            if urls:
                return {"photos": urls, "source": "foursquare"}
        except Exception:
            pass

    if lat is None or lng is None:
        geo = geocode_location.invoke({"place_name": place_name})
        lat, lng = geo["lat"], geo["lng"]
    static = get_static_map_url.invoke({"lat": lat, "lng": lng})
    if static.get("url"):
        return {"photos": [static["url"]], "source": static["source"]}
    return {"photos": [], "source": "none"}


@tool(parse_docstring=True)
def check_attraction_availability(activity_id: str, destination: str) -> dict:
    """Check whether an attraction appears open (Foursquare hours / seed hours).

    Args:
        activity_id: Activity id.
        destination: City the activity belongs to.

    Returns:
        dict with available (bool), opening_hour, closing_hour, source.
    """
    hours = get_opening_hours.invoke({"activity_id": activity_id, "destination": destination})
    open_h, close_h = hours["opening_hour"], hours["closing_hour"]
    return {
        "available": close_h - open_h >= 1,
        "opening_hour": open_h,
        "closing_hour": close_h,
        "source": hours.get("source", "default"),
    }


__all__ = [
    "search_destinations",
    "search_attractions",
    "get_opening_hours",
    "get_place_photos",
    "check_attraction_availability",
]
