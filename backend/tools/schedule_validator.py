"""Overlap and day-length checks on a finished itinerary (used by the Critic)."""
from __future__ import annotations


def check_schedule_conflicts(days: list[dict]) -> dict:
    """Every day: no two items overlap, and no day spans more than 14 hours."""
    issues: list[dict] = []
    for day in days:
        items = sorted(day["items"], key=lambda i: i["start_hour"])
        for current, nxt in zip(items, items[1:]):
            if current["end_hour"] > nxt["start_hour"]:
                issues.append({"day": day["day_number"], "message": f"Overlap: '{current['activity_name']}' ends at {current['end_hour']:.1f} but '{nxt['activity_name']}' starts at {nxt['start_hour']:.1f}"})
        if items:
            span = max(i["end_hour"] for i in items) - min(i["start_hour"] for i in items)
            if span > 14:
                issues.append({"day": day["day_number"], "message": f"Day {day['day_number']} spans {span:.1f}h — likely overpacked"})
    return {"valid": not issues, "issues": issues}
