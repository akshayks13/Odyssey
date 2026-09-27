"""The offline world: cities, sights, hotels, road and rail networks, and monthly weather.

Everything the agents know comes from one JSON file per region in backend/data/. There are no API
calls, so the same request always produces the same plan. A new region is a new JSON file.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from algorithms.search import build_graph, uniform_cost_search

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CITY_SPEED_KMH = 25.0  # getting around between sights inside a city
HOP_OVERHEAD_MIN = 10  # parking, walking in
RAINY_CHANCE = 60  # % chance of rain from which a day counts as rainy


@lru_cache(maxsize=None)
def regions() -> dict[str, dict]:
    """Every region in data/, keyed by its name."""
    out = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        out[data["region"]] = data
    return out


def match_region(text: str) -> str | None:
    """The region a request names, directly or through one of its cities."""
    low = text.lower()
    for name, data in regions().items():
        words = [name.split(",")[0].lower(), *data.get("aliases", [])]
        for city in data["cities"]:
            words += [city["name"].lower(), *city.get("aliases", [])]
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
            return name
    return None


def world(region: str) -> dict:
    return regions()[region]


def cities(region: str) -> list[dict]:
    return world(region)["cities"]


def city(region: str, name: str) -> dict | None:
    return next((c for c in cities(region) if c["name"] == name), None)


def match_city(region: str, text: str) -> str | None:
    """A city name (or alias) as written by the traveller -> the dataset's name for it."""
    low = text.strip().lower()
    for c in cities(region):
        if low == c["name"].lower() or low in c.get("aliases", []):
            return c["name"]
    return None


def cities_named_in(region: str, text: str) -> list[str]:
    """Cities mentioned anywhere in `text`, in the order they appear."""
    low = text.lower()
    hits = []
    for c in cities(region):
        positions = [m.start() for w in [c["name"].lower(), *c.get("aliases", [])] for m in re.finditer(rf"\b{re.escape(w)}\b", low)]
        if positions:
            hits.append((min(positions), c["name"]))
    return [name for _, name in sorted(hits)]


def sights(region: str, city_name: str) -> list[dict]:
    return world(region)["sights"].get(city_name, [])


def hotels(region: str, city_name: str) -> list[dict]:
    return world(region)["hotels"].get(city_name, [])


def food_per_day(region: str, tier: str) -> float:
    return float(world(region)["food_per_person_per_day_inr"].get(tier, 900))


# ---------------------------------------------------------------------------
# Travel between cities: UCS over the road and rail networks
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _road_graph(region: str):
    return build_graph(world(region)["roads"], "hours", "km")


@lru_cache(maxsize=None)
def _rail_graph(region: str):
    return build_graph(world(region)["rail"], "hours", "fare_inr")


@lru_cache(maxsize=None)
def road_route(region: str, a: str, b: str) -> dict:
    """Fastest road path (UCS): hours, km, the towns it passes, nodes expanded."""
    r = uniform_cost_search(_road_graph(region), a, b)
    return {"found": r["found"], "hours": r["hours"], "km": r["extra"], "path": r["path"], "nodes_expanded": r["nodes_expanded"]}


@lru_cache(maxsize=None)
def rail_route(region: str, a: str, b: str) -> dict:
    """Fastest train (UCS on the rail network). Only between towns that have a station."""
    r = uniform_cost_search(_rail_graph(region), a, b)
    return {"found": r["found"], "hours": r["hours"], "fare_inr": r["extra"], "path": r["path"], "nodes_expanded": r["nodes_expanded"]}


# ---------------------------------------------------------------------------
# Distances inside a city, and weather
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_time_matrix(points: list[dict]) -> list[list[int]]:
    """Minutes between every pair of points: straight-line distance at city speed, plus a fixed overhead."""
    out = []
    for a in points:
        row = []
        for b in points:
            if a is b:
                row.append(0)
                continue
            km = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
            row.append(int(round(km / CITY_SPEED_KMH * 60)) + HOP_OVERHEAD_MIN)
        out.append(row)
    return out


def weather_on(region: str, city_name: str, day: date) -> dict:
    """Typical weather for that city in that month (climate averages, not a forecast)."""
    data = world(region)
    base = data["weather_by_month"][str(day.month)]
    c = city(region, city_name) or {}
    adjust = data.get("hill_adjustment", {}) if c.get("hill") else {}
    tmin = base["tmin"] + adjust.get("temp_c", 0)
    tmax = base["tmax"] + adjust.get("temp_c", 0)
    chance = min(100, base["rain_chance"] + adjust.get("rain_chance", 0))
    return {
        "date": day.isoformat(),
        "condition": base["condition"],
        "tmin": float(tmin),
        "tmax": float(tmax),
        "rain_mm": float(base["rain_mm"]),
        "rain_chance": int(chance),
        "rainy": chance >= RAINY_CHANCE,
        "source": "monthly average",
    }
