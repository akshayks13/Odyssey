"""Mapbox geocoding, driving directions, and travel-time matrices."""
from __future__ import annotations

import math

import httpx
from langchain_core.tools import tool

from config import MAPBOX_API_KEY
from tools.seed_data import get_destination_seed, get_travel_leg

MAPBOX_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving/{coords}"
MAPBOX_MATRIX_URL = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords}"
MAPBOX_STATIC_URL = "https://api.mapbox.com/styles/v1/mapbox/outdoors-v12/static/{lng},{lat},12,0/600x400"

_AVG_ROAD_SPEED_KMH = 40.0  # used when we only have haversine distance

_KNOWN_COORDS = {
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "goa": (15.2993, 74.1240),
    "kochi": (9.9312, 76.2673),
    "cochin": (9.9312, 76.2673),
    "trivandrum": (8.5241, 76.9366),
    "thiruvananthapuram": (8.5241, 76.9366),
}


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
        dict with keys: name, lat, lng, source ("mapbox" or "seed").
    """
    if MAPBOX_API_KEY:
        try:
            resp = httpx.get(
                MAPBOX_GEOCODE_URL.format(query=place_name),
                params={"access_token": MAPBOX_API_KEY, "limit": 1},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("features"):
                feat = data["features"][0]
                lng, lat = feat["center"]
                return {"name": feat.get("text", place_name), "lat": lat, "lng": lng, "source": "mapbox"}
        except Exception:
            pass

    city_key = place_name.split(",")[0].strip()
    seed = get_destination_seed(place_name) or get_destination_seed(city_key)
    if seed:
        return {
            "name": seed["name"],
            "lat": seed["coordinates"]["lat"],
            "lng": seed["coordinates"]["lng"],
            "source": "seed",
        }
    known = _KNOWN_COORDS.get(place_name.strip().lower()) or _KNOWN_COORDS.get(city_key.lower())
    if known:
        lat, lng = known
        return {"name": city_key, "lat": lat, "lng": lng, "source": "known_city"}

    return {"name": place_name, "lat": 21.1466, "lng": 79.0889, "source": "approx"}


@tool(parse_docstring=True)
def get_directions(origin: str, destination: str) -> dict:
    """Get road travel time, distance and cost between two named places.

    Args:
        origin: Name of the origin destination (e.g. "Kochi").
        destination: Name of the destination (e.g. "Munnar").

    Returns:
        dict with keys: distance_km, duration_hours, cost_inr, mode, source.
    """
    if MAPBOX_API_KEY:
        try:
            o = geocode_location.invoke({"place_name": origin})
            d = geocode_location.invoke({"place_name": destination})
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
                return {
                    "distance_km": round(distance_km, 1),
                    "duration_hours": round(duration_hours, 2),
                    "cost_inr": round(distance_km * 7.0, 0),  # ~INR 7/km taxi estimate
                    "mode": "road",
                    "source": "mapbox",
                }
        except Exception:
            pass

    leg = get_travel_leg(origin, destination)
    if leg:
        return {
            "distance_km": leg["distance_km"],
            "duration_hours": leg["duration_hours"],
            "cost_inr": leg["cost_inr"],
            "mode": leg.get("mode", "road"),
            "source": "seed",
        }

    o = geocode_location.invoke({"place_name": origin})
    d = geocode_location.invoke({"place_name": destination})
    dist = haversine_km(o["lat"], o["lng"], d["lat"], d["lng"])
    duration = dist / _AVG_ROAD_SPEED_KMH
    return {
        "distance_km": round(dist, 1),
        "duration_hours": round(duration, 2),
        "cost_inr": round(dist * 7.0, 0),
        "mode": "road",
        "source": "haversine_estimate",
    }


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
    """Build a square travel-time matrix (minutes) for VRPTW using Mapbox Matrix.

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

    if MAPBOX_API_KEY and len(points) <= 25:
        try:
            coords = ";".join(f"{p['lng']},{p['lat']}" for p in points)
            resp = httpx.get(
                MAPBOX_MATRIX_URL.format(coords=coords),
                params={"access_token": MAPBOX_API_KEY, "annotations": "duration"},
                timeout=8.0,
            )
            resp.raise_for_status()
            durations = resp.json().get("durations")
            if durations:
                matrix = [[max(0, int(round((cell or 0) / 60))) for cell in row] for row in durations]
                return {"matrix": matrix, "source": "mapbox"}
        except Exception:
            pass

    return {"matrix": _haversine_matrix(points), "source": "haversine"}


@tool(parse_docstring=True)
def get_static_map_url(lat: float, lng: float) -> dict:
    """Mapbox Static Images URL for a coordinate (used as a photo fallback).

    Args:
        lat: Latitude.
        lng: Longitude.

    Returns:
        dict with url and source.
    """
    if not MAPBOX_API_KEY:
        return {"url": None, "source": "none"}
    url = MAPBOX_STATIC_URL.format(lng=lng, lat=lat) + f"?access_token={MAPBOX_API_KEY}"
    return {"url": url, "source": "mapbox_static"}
