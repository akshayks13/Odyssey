"""Foursquare Places tools: attraction search, opening hours, photos.

Used by:
- Destination Agent (search_destinations, search_attractions, get_place_photos)
- Itinerary Architect (get_opening_hours → Places Details)
- Critic (check_attraction_availability)

Live path: Foursquare Places Search + Place Details when FOURSQUARE_API_KEY
is set (Places API host `places-api.foursquare.com`, Bearer service key).
Falls back to the Kerala seed dataset when the key is missing or
the API call fails.
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config import FOURSQUARE_API_KEY, MAPBOX_API_KEY
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
    cat_l = cat_name.lower()
    mapped = "general"
    for key in ("nature", "adventure", "food", "nightlife", "culture", "shopping", "relax"):
        if key in cat_l:
            mapped = "relaxation" if key == "relax" else key
            break
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


@tool(parse_docstring=True)
def search_destinations(region: str) -> list[dict]:
    """Find candidate destinations (cities/towns) within a region.

    Tries Mapbox geocoding + Foursquare Places search first, then Kerala seed.

    Args:
        region: Broad region name, e.g. "Kerala, India".

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

    live: list[dict] = []
    geo = geocode_location.invoke({"place_name": region})
    headers = _fsq_headers()
    if headers:
        try:
            resp = httpx.get(
                FSQ_SEARCH_URL,
                headers=headers,
                params={"near": region, "query": f"{key} town viewpoint park beach", "limit": 10},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            for place in resp.json().get("results") or []:
                lat, lng = _coords_from_place(place)
                name = place.get("name")
                if not name or lat is None:
                    continue
                live.append(
                    {
                        "name": name,
                        "lat": lat,
                        "lng": lng,
                        "description": (place.get("location") or {}).get("formatted_address", region),
                        "tags": [c.get("name", "") for c in (place.get("categories") or []) if c.get("name")],
                        "activity_scores": dict(_DEFAULT_SCORES),
                        "source": "foursquare",
                    }
                )
        except Exception:
            live = []

    if MAPBOX_API_KEY and geo.get("source") == "mapbox" and not any(d["name"].lower() == key for d in seed_hits + live):
        live.append(
            {
                "name": geo.get("name") or region.split(",")[0].strip(),
                "lat": geo["lat"],
                "lng": geo["lng"],
                "description": f"Geocoded region center for {region}",
                "tags": [],
                "activity_scores": dict(_DEFAULT_SCORES),
                "source": "mapbox",
            }
        )

    # Prefer curated seed cities when they match (highest-quality Kerala demo),
    # then append unique live hits so a non-Kerala region still works with keys.
    by_name: dict[str, dict] = {}
    for row in [*seed_hits, *live]:
        by_name.setdefault(row["name"].lower(), row)
    results = list(by_name.values())
    if results:
        return results
    return [
        {
            "name": geo.get("name") or region.split(",")[0].strip(),
            "lat": geo["lat"],
            "lng": geo["lng"],
            "description": f"Fallback geocode for {region}",
            "tags": [],
            "activity_scores": dict(_DEFAULT_SCORES),
            "source": geo.get("source", "seed_default"),
        }
    ]


@tool(parse_docstring=True)
def search_attractions(destination: str, category: str | None = None) -> list[dict]:
    """Search for attractions/activities in a specific destination.

    Args:
        destination: Name of the destination, e.g. "Munnar".
        category: Optional category filter, e.g. "nature" or "adventure".

    Returns:
        List of activity dicts with id, name, category, duration_minutes,
        cost_inr, rating, opening_hour, closing_hour, preference_score.
    """
    headers = _fsq_headers()
    if headers:
        try:
            params: dict = {"near": f"{destination}, India", "limit": 15}
            if category:
                params["query"] = category
            resp = httpx.get(FSQ_SEARCH_URL, headers=headers, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if results:
                out = [_place_to_activity(r, destination, category) for r in results]
                for act in out[:8]:
                    _enrich_hours(act)
                return out
        except Exception:
            pass

    activities = get_activities(destination)
    if category:
        activities = [a for a in activities if a["category"] == category] or activities
    dest = next((d for d in all_destinations() if d["name"].lower() == destination.lower()), None)
    coords = dest["coordinates"] if dest else None
    out = []
    for a in activities:
        row = {**a, "destination": destination, "source": "seed"}
        if coords and not row.get("coordinates"):
            row["coordinates"] = coords
        out.append(row)
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


# Keep load_seed imported for region filtering helpers used in tests.
__all__ = [
    "search_destinations",
    "search_attractions",
    "get_opening_hours",
    "get_place_photos",
    "check_attraction_availability",
]
