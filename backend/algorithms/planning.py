"""Trip-shape helpers shared by Budget and the Itinerary Architect: days per city, and skipped activities."""
from __future__ import annotations

import math

from models.schemas import EditLocks, StayBlock

PACE_CAPS = {"relaxed": 2, "moderate": 3, "packed": 4}
PACE_HOURS = {"relaxed": (9.0, 18.0), "moderate": (8.0, 21.0), "packed": (7.0, 22.0)}

TRAVEL_DAY_HOURS = 8.0  # a transfer this long uses up the whole day


def plan_stays(order: list[str], duration: int, inbound_hours: dict[str, float], activity_counts: dict[str, int]) -> list[StayBlock]:
    """Split `duration` days across the ordered cities.

    Each city gets one day; a city reached by a transfer of TRAVEL_DAY_HOURS or more spends its first
    day travelling. Extra days go to the cities with the most sights left (about 2 sights a day).
    Nights = days, minus one for the last city. Budget bills these nights and the Architect schedules
    these days, so the two always agree.
    """
    if not order or duration <= 0:
        return []
    order = order[:duration]
    travel_only = {c: inbound_hours.get(c, 0.0) >= TRAVEL_DAY_HOURS for c in order}
    days = {c: 1 + (1 if travel_only[c] else 0) for c in order}
    if sum(days.values()) > duration:  # no room for the transfer days
        travel_only = {c: False for c in order}
        days = {c: 1 for c in order}

    sights = {c: max(1, activity_counts.get(c, 1)) for c in order}
    for _ in range(duration - sum(days.values())):
        def room(c: str) -> float:
            sight_days = days[c] - (1 if travel_only[c] else 0)
            return sights[c] / (sight_days + 1) if sight_days < math.ceil(sights[c] / 2) else -1.0
        days[max(order, key=room)] += 1

    return [
        StayBlock(destination=c, days=days[c], nights=max(0, days[c] - (1 if i == len(order) - 1 else 0)), travel_day=travel_only[c])
        for i, c in enumerate(order)
    ]


def activity_is_excluded(name: str, locks: EditLocks | None) -> bool:
    """True if the user asked to skip this activity ("the museum" matches "City Museum")."""
    if not locks:
        return False
    low = name.lower()
    return any(x.lower() in low or low in x.lower() for x in locks.excluded_activities if x.strip())
