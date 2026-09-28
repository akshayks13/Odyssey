"""One day's timetable as a constraint satisfaction problem, solved by backtracking search.

Formulation:
    variables    one per item: each chosen sight, plus lunch and dinner
    domains      start times in 15-minute steps: after the sight opens and after the traveller can get
                 there from the hotel, and early enough to finish before it closes and before the day ends
                 (lunch 12:00-14:00, dinner 19:00-21:00)
    constraints  every pair (X, Y) must not overlap, with travel time between them:
                 start(Y) >= end(X) + travel(X, Y)  or  start(X) >= end(Y) + travel(Y, X)

Search: backtracking with
    MRV            pick the unassigned variable with the fewest values left
    value order    earliest start first, so the day stays compact
    forward check  after each assignment, remove clashing values from every other domain and
                   backtrack at once if a domain becomes empty
If the full set of sights cannot fit, the least-wanted sight is dropped and the CSP is solved again.
"""
from __future__ import annotations

import math

STEP = 15  # minutes per slot
MEALS = (("_meal_lunch", "Lunch", 12.0, 14.0, 45), ("_meal_dinner", "Dinner", 19.0, 21.0, 45))
DEFAULT_HOP_MINUTES = 10
MAX_DEPOT_MINUTES = 45


class _Solver:
    def __init__(self, items: list[dict], travel: list[list[int]]):
        self.items = items
        self.travel = travel
        self.nodes = 0
        self.backtracks = 0

    def clash(self, x: int, sx: int, y: int, sy: int) -> bool:
        ex = sx + self.items[x]["dur"]
        ey = sy + self.items[y]["dur"]
        return not (sy >= ex + self.travel[x][y] or sx >= ey + self.travel[y][x])

    def solve(self, domains: dict[int, list[int]], assignment: dict[int, int]) -> dict[int, int] | None:
        if len(assignment) == len(self.items):
            return assignment
        unassigned = [v for v in domains if v not in assignment]
        var = min(unassigned, key=lambda v: (len(domains[v]), v))  # MRV, ties by index (deterministic)
        for value in domains[var]:
            self.nodes += 1
            pruned: dict[int, list[int]] = {}
            wiped_out = False
            for other in unassigned:
                if other == var:
                    continue
                kept = [t for t in domains[other] if not self.clash(var, value, other, t)]
                if not kept:
                    wiped_out = True
                    break
                pruned[other] = kept
            if not wiped_out:
                result = self.solve({**domains, **pruned, var: [value]}, {**assignment, var: value})
                if result is not None:
                    return result
            self.backtracks += 1
        return None


def _first_fit(items: list[dict], travel: list[list[int]], domains: dict[int, list[int]], n_sights: int, stats: dict) -> dict[int, int]:
    """The baseline: greedy first-fit with a simple repair.

    Sights (already sorted by preference) each take the earliest start that clashes with nothing placed so far, with no
    look-ahead. Then each meal takes its earliest free slot; if it has none, the least-wanted placed sight is dropped and
    the meal tries again. So it also honours meals and travel time, but it never re-arranges what it has placed."""
    solver = _Solver(items, travel)

    def earliest(i: int, placed: dict[int, int]) -> int | None:
        for start in domains[i]:
            stats["nodes"] += 1
            if all(not solver.clash(i, start, j, s) for j, s in placed.items()):
                return start
        return None

    placed: dict[int, int] = {}
    for i in range(n_sights):
        start = earliest(i, placed)
        if start is not None:
            placed[i] = start
    for meal in range(n_sights, len(items)):
        start = earliest(meal, placed)
        while start is None and any(i < n_sights for i in placed):
            del placed[max(i for i in placed if i < n_sights)]  # the least wanted sight goes
            start = earliest(meal, placed)
        if start is not None:
            placed[meal] = start
    return placed


def _domain(item: dict, day_start: int, day_end: int, from_hotel: int) -> list[int]:
    earliest = max(day_start + from_hotel, item["open"])
    latest = min(item["close"], day_end) - item["dur"]
    first = math.ceil(earliest / STEP) * STEP
    return list(range(first, latest + 1, STEP))


def solve_day_schedule(
    activities: list[dict],
    day_start_hour: float = 8.0,
    day_end_hour: float = 21.0,
    travel_matrix_minutes: list[list[int]] | None = None,
    include_meals: bool = True,
    method: str = "csp",
) -> dict:
    """Timetable for one day.

    Args:
        method: "csp" (backtracking + MRV + forward checking) or "greedy" (first-fit baseline).
        activities: dicts with id, name, duration_minutes, opening_hour, closing_hour,
            preference_score, category.
        day_start_hour / day_end_hour: the usable window (later on an arrival day).
        travel_matrix_minutes: square matrix over [hotel] + activities. Missing: a short default hop.
        include_meals: add lunch and dinner when their window falls inside the day.

    Returns:
        scheduled (sights with start/end hours), meals, selected_ids, status, and stats
        (nodes tried, backtracks, sights dropped to make the day fit).
    """
    stats = {"nodes": 0, "backtracks": 0, "dropped": 0}
    if not activities:
        return {"scheduled": [], "meals": [], "selected_ids": [], "status": "NO_ACTIVITIES", "stats": stats}

    day_start, day_end = int(day_start_hour * 60), int(day_end_hour * 60)
    n = len(activities)
    base = travel_matrix_minutes if travel_matrix_minutes and len(travel_matrix_minutes) == n + 1 else None

    sights = [
        {
            "idx": i + 1,  # row in the travel matrix (0 is the hotel)
            "id": a["id"],
            "name": a["name"],
            "category": a.get("category", "general"),
            "pref": float(a.get("preference_score") or 0.5),
            "dur": max(15, int(a["duration_minutes"])),
            "open": int(float(a["opening_hour"]) * 60),
            "close": int(float(a["closing_hour"]) * 60),
        }
        for i, a in enumerate(activities)
    ]
    meals = []
    if include_meals:
        for mid, name, open_h, close_h, dur in MEALS:
            if open_h * 60 >= day_start and close_h * 60 <= day_end + 60:
                meals.append({"idx": None, "id": mid, "name": name, "category": "meal", "pref": 0.0, "dur": dur,
                              "open": int(open_h * 60), "close": int(close_h * 60), "meal": True})

    def hop(a: dict, b: dict) -> int:
        if a.get("meal") or b.get("meal") or base is None:
            return DEFAULT_HOP_MINUTES
        return int(base[a["idx"]][b["idx"]])

    def from_hotel(item: dict) -> int:
        if item.get("meal") or base is None:
            return 0
        return min(int(base[0][item["idx"]]), MAX_DEPOT_MINUTES)

    # Most-wanted first; when the day cannot hold them all, the last one is dropped and we try again.
    candidates = sorted(sights, key=lambda s: -s["pref"])
    if method == "greedy":
        items = candidates + meals
        travel = [[0 if i == j else hop(a, b) for j, b in enumerate(items)] for i, a in enumerate(items)]
        domains = {i: _domain(item, day_start, day_end, from_hotel(item)) for i, item in enumerate(items)}
        result = _first_fit(items, travel, domains, len(candidates), stats)
        stats["dropped"] = sum(1 for i in range(len(candidates)) if i not in result)
    else:
        while True:
            items = candidates + meals
            travel = [[0 if i == j else hop(a, b) for j, b in enumerate(items)] for i, a in enumerate(items)]
            domains = {i: _domain(item, day_start, day_end, from_hotel(item)) for i, item in enumerate(items)}
            solver = _Solver(items, travel)
            result = None if any(not d for d in domains.values()) else solver.solve(domains, {})
            stats["nodes"] += solver.nodes
            stats["backtracks"] += solver.backtracks
            if result is not None:
                break
            if candidates:
                candidates = candidates[:-1]
                stats["dropped"] += 1
            elif meals:
                meals = []  # not even the meals fit this window (a late arrival): schedule nothing
            else:
                return {"scheduled": [], "meals": [], "selected_ids": [], "status": "INFEASIBLE", "stats": stats}

    rows = sorted(
        ({"id": item["id"], "name": item["name"], "category": item["category"], "meal": item.get("meal", False),
          "start_hour": round(start / 60, 2), "end_hour": round((start + item["dur"]) / 60, 2)}
         for item, start in ((items[i], s) for i, s in result.items())),
        key=lambda r: r["start_hour"],
    )
    scheduled = [{k: v for k, v in r.items() if k != "meal"} for r in rows if not r["meal"]]
    meal_rows = [{k: v for k, v in r.items() if k != "meal"} for r in rows if r["meal"]]
    return {
        "scheduled": scheduled,
        "meals": meal_rows,
        "selected_ids": [s["id"] for s in scheduled],
        "status": "FEASIBLE" if scheduled else "EMPTY",
        "stats": stats,
    }
