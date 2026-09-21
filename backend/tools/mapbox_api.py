"""Places and travel times. Mapbox first; OpenStreetMap (Nominatim for places, OSRM for roads) if Mapbox has no answer."""
from __future__ import annotations

import math

import httpx
from langchain_core.tools import tool

from config import MAPBOX_API_KEY
from tools.cost_calculator import ROAD_VEHICLES

MAPBOX_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving/{coords}"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"

_geocode_cache: dict[str, dict] = {}
_directions_cache: dict[tuple[str, str], dict] = {}

_OWN_CAR_PER_KM = ROAD_VEHICLES["own_car"]["per_km"]
_OSRM_REAL_ROADS = 1.4  # OSRM assumes free-flowing roads; on Indian roads a drive takes about 1.4x as long


def _nominatim(place_name: str) -> dict | None:
    """OpenStreetMap's free geocoder: the fallback when Mapbox has no answer."""
    try:
        resp = httpx.get(
            NOMINATIM_URL,
            params={"q": place_name, "format": "json", "limit": 1},
            headers={"User-Agent": "Odyssey-Travel-Planner/1.0"},
            timeout=8.0,
        )
        resp.raise_for_status()
        hit = (resp.json() or [None])[0]
        if hit:
            return {"name": place_name.split(",")[0].strip(), "lat": float(hit["lat"]), "lng": float(hit["lon"]), "source": "osm"}
    except Exception:  # noqa: BLE001
        pass
    return None


def _osrm(o: dict, d: dict) -> dict | None:
    """OpenStreetMap road distance and time between two points (free, no key)."""
    try:
        resp = httpx.get(
            OSRM_URL.format(coords=f"{o['lng']},{o['lat']};{d['lng']},{d['lat']}"),
            params={"overview": "false"},
            timeout=8.0,
        )
        resp.raise_for_status()
        route = resp.json()["routes"][0]
        km = route["distance"] / 1000.0
        return {
            "distance_km": round(km, 1),
            "duration_hours": round(route["duration"] / 3600.0 * _OSRM_REAL_ROADS, 2),
            "cost_inr": round(km * _OWN_CAR_PER_KM, 0),
            "mode": "road",
            "source": "osrm",
        }
    except Exception:  # noqa: BLE001
        return None


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@tool(parse_docstring=True)
def geocode_location(place_name: str) -> dict:
    """Resolve a free-text place name to coordinates and a canonical name.

    Args:
        place_name: Free-text location, e.g. "Kerala" or "Munnar, India".

    Returns:
        dict with keys: name, lat, lng, source ("mapbox" or "osm"). lat and lng are None when the place cannot be found.
    """
    cache_key = place_name.strip().lower()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]
    found = None
    if MAPBOX_API_KEY:
        try:
            resp = httpx.get(
                MAPBOX_GEOCODE_URL.format(query=place_name),
                params={"access_token": MAPBOX_API_KEY, "limit": 1},
                timeout=5.0,
            )
            resp.raise_for_status()
            features = resp.json().get("features")
            if features:
                lng, lat = features[0]["center"]
                found = {"name": features[0].get("text", place_name), "lat": lat, "lng": lng, "source": "mapbox"}
        except Exception:  # noqa: BLE001
            pass
    found = found or _nominatim(place_name)  # the one fallback
    if not found:
        return {"name": place_name, "lat": None, "lng": None, "source": "not_found"}
    _geocode_cache[cache_key] = found
    return found


@tool(parse_docstring=True)
def get_directions(origin: str, destination: str) -> dict:
    """Get road travel time, distance and cost between two named places.

    Args:
        origin: Name of the origin destination (e.g. "Kochi").
        destination: Name of the destination (e.g. "Munnar").

    Returns:
        dict with keys: distance_km, duration_hours, cost_inr, mode, source ("mapbox", "osrm" or "unavailable").
    """
    cache_key = (origin.strip().lower(), destination.strip().lower())
    if cache_key in _directions_cache:
        return _directions_cache[cache_key]
    if MAPBOX_API_KEY:
        try:
            o = geocode_location.invoke({"place_name": origin})
            d = geocode_location.invoke({"place_name": destination})
            if o["lat"] is None or d["lat"] is None:
                raise ValueError("place not found")
            coords = f"{o['lng']},{o['lat']};{d['lng']},{d['lat']}"
            resp = httpx.get(
                MAPBOX_DIRECTIONS_URL.format(coords=coords),
                params={"access_token": MAPBOX_API_KEY, "overview": "false"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("routes"):
                route = data["routes"][0]
                distance_km = route["distance"] / 1000.0
                duration_hours = route["duration"] / 3600.0
                found = {
                    "distance_km": round(distance_km, 1),
                    "duration_hours": round(duration_hours, 2),
                    "cost_inr": round(distance_km * _OWN_CAR_PER_KM, 0),  # one own car; Mobility prices the actual vehicle
                    "mode": "road",
                    "source": "mapbox",
                }
                _directions_cache[cache_key] = found
                return found
        except Exception:
            pass

    o = geocode_location.invoke({"place_name": origin})
    d = geocode_location.invoke({"place_name": destination})
    found = _osrm(o, d) if o["lat"] is not None and d["lat"] is not None else None
    if found:
        _directions_cache[cache_key] = found
        return found
    return {"distance_km": 0.0, "duration_hours": 0.0, "cost_inr": 0.0, "mode": "road", "source": "unavailable"}


def _haversine_matrix(points: list[dict]) -> list[list[int]]:
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = haversine_km(points[i]["lat"], points[i]["lng"], points[j]["lat"], points[j]["lng"])
            # Intra-city / local hops: ~25 km/h average including traffic.
            matrix[i][j] = max(5, int(round(dist / 25.0 * 60)))
    return matrix


@tool(parse_docstring=True)
def travel_time_matrix(points: list[dict]) -> dict:
    """Square travel-time matrix (minutes) between a day's stops, for the scheduler. Stops in one city are
    a few km apart, so this uses straight-line distance at city speed instead of a routing service.

    Args:
        points: Ordered list of {lat, lng} dicts. Index 0 should be the depot
            (hotel / city center); remaining points are that day's activities.

    Returns:
        dict with matrix (list[list[int]] minutes) and source.
    """
    if not points:
        return {"matrix": [], "source": "empty"}
    if len(points) == 1:
        return {"matrix": [[0]], "source": "trivial"}
    return {"matrix": _haversine_matrix(points), "source": "haversine"}
