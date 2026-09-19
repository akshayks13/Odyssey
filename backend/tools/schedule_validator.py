"""Deterministic schedule validation — detects time-window and overlap
conflicts in a constructed itinerary. Used by the Itinerary Architect
(post-solve check) and the Critic (independent verification).
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool(parse_docstring=True)
def validate_time_windows(scheduled_items: list[dict]) -> dict:
    """Check a list of scheduled items (same day) for overlaps or gaps that
    violate a maximum idle threshold.

    Args:
        scheduled_items: List of dicts each with start_hour, end_hour,
            activity_name.

    Returns:
        dict with valid (bool) and conflicts (list of human-readable strings).
    """
    conflicts: list[str] = []
    items = sorted(scheduled_items, key=lambda i: i["start_hour"])
    for i in range(len(items) - 1):
        current, nxt = items[i], items[i + 1]
        if current["end_hour"] > nxt["start_hour"]:
            conflicts.append(
                f"Overlap: '{current['activity_name']}' ends at {current['end_hour']:.1f} "
                f"but '{nxt['activity_name']}' starts at {nxt['start_hour']:.1f}"
            )
    return {"valid": len(conflicts) == 0, "conflicts": conflicts}


@tool(parse_docstring=True)
def check_schedule_conflicts(days: list[dict]) -> dict:
    """Validate every day of a multi-day itinerary for overlaps and excessive
    daily duration.

    Args:
        days: List of day dicts, each with day_number and items (list of
            dicts with start_hour, end_hour, activity_name).

    Returns:
        dict with valid (bool) and issues (list of dicts: day, message).
    """
    issues: list[dict] = []
    for day in days:
        result = validate_time_windows.invoke({"scheduled_items": day["items"]})
        for conflict in result["conflicts"]:
            issues.append({"day": day["day_number"], "message": conflict})

        if day["items"]:
            span = max(i["end_hour"] for i in day["items"]) - min(i["start_hour"] for i in day["items"])
            if span > 14:
                issues.append(
                    {
                        "day": day["day_number"],
                        "message": f"Day {day['day_number']} spans {span:.1f}h — likely overpacked",
                    }
                )
    return {"valid": len(issues) == 0, "issues": issues}
