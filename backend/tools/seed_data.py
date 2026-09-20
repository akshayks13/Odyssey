"""Loader for `data/kerala_seed.json`.

Used when a live lookup has no row for that city (missing key, failed
request, or a city that is actually in the bundled set).
"""
from __future__ import annotations

import functools
import json

from config import SEED_DATA_PATH


@functools.lru_cache(maxsize=1)
def load_seed() -> dict:
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_destination_seed(name: str) -> dict | None:
    for dest in load_seed()["destinations"]:
        if dest["name"].lower() == name.lower():
            return dest
    return None


def all_destinations() -> list[dict]:
    seed = load_seed()
    region = seed.get("region", "Kerala, India")
    return [{**d, "region": d.get("region", region)} for d in seed["destinations"]]


def get_activities(destination: str) -> list[dict]:
    return load_seed()["activities"].get(destination, [])


def get_hotels(destination: str) -> list[dict]:
    return load_seed()["hotels"].get(destination, [])


def get_food_cost(tier: str = "mid") -> float:
    return load_seed()["food_cost_per_person_per_day_inr"].get(tier, 900)


def get_travel_leg(origin: str, destination: str) -> dict | None:
    legs = load_seed()["travel_legs"]
    for leg in legs:
        if leg["origin"] == origin and leg["destination"] == destination:
            return leg
        if leg["origin"] == destination and leg["destination"] == origin:
            # reverse — same cost/time assumed both directions
            return {**leg, "origin": origin, "destination": destination}
    return None
