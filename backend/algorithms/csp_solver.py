"""OR-Tools VRPTW scheduler for a single day."""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp

_LUNCH = ("_meal_lunch", "Lunch", 12.0, 14.0, 45)
_DINNER = ("_meal_dinner", "Dinner", 19.0, 21.0, 45)
_DEFAULT_HOP_MINUTES = 12
_MAX_DEPOT_MINUTES = 45  # the city centre stands in for the hotel; don't let a far-off centre block every day


def _default_matrix(n: int) -> list[list[int]]:
    return [[0 if i == j else _DEFAULT_HOP_MINUTES for j in range(n)] for i in range(n)]


def _expand_matrix(base: list[list[int]], extra: int) -> list[list[int]]:
    n = len(base) + extra
    matrix = [row[:] + [_DEFAULT_HOP_MINUTES] * extra for row in base]
    matrix += [[_DEFAULT_HOP_MINUTES] * n for _ in range(extra)]
    for i in range(n):
        matrix[i][i] = 0
    return matrix


def solve_day_schedule(
    activities: list[dict],
    day_start_hour: float = 8.0,
    day_end_hour: float = 21.0,
    travel_matrix_minutes: list[list[int]] | None = None,
    include_meals: bool = True,
    max_time_in_seconds: float = 0.4,
) -> dict:
    """Vehicle Routing with Time Windows for a single day / single vehicle.

    Args:
        activities: dicts with id, name, duration_minutes, opening_hour,
            closing_hour, preference_score, category.
        day_start_hour: earliest start (later on an arrival day).
        day_end_hour: latest return to depot.
        travel_matrix_minutes: square matrix over [depot] + activities.
            If omitted, a small intra-city default is used.
        include_meals: add lunch/dinner nodes with time windows.

    Returns:
        scheduled (real activities), meals, selected_ids, status.
    """
    if not activities:
        return {"scheduled": [], "meals": [], "selected_ids": [], "status": "NO_ACTIVITIES"}

    day_start = int(day_start_hour * 60)
    day_end = int(day_end_hour * 60)

    # Keep activities that can fit their opening window; remap the travel matrix to match.
    keep: list[int] = []
    feasible: list[dict] = []
    for i, act in enumerate(activities):
        dur = max(15, int(act["duration_minutes"]))
        open_m = max(day_start, int(float(act["opening_hour"]) * 60))
        close_m = min(day_end, int(float(act["closing_hour"]) * 60))
        if close_m - open_m < dur:
            continue
        keep.append(i)
        feasible.append({**act, "duration_minutes": dur, "_open": open_m, "_close": close_m - dur})
    if not feasible:
        return {"scheduled": [], "meals": [], "selected_ids": [], "status": "INFEASIBLE"}

    if travel_matrix_minutes and len(travel_matrix_minutes) == len(activities) + 1:
        rows = [0] + [i + 1 for i in keep]
        base = [[int(travel_matrix_minutes[r][c]) for c in rows] for r in rows]
    else:
        base = _default_matrix(len(feasible) + 1)

    meal_nodes: list[dict] = []
    if include_meals:
        for mid, name, open_h, close_h, dur in (_LUNCH, _DINNER):
            open_m, close_m = int(open_h * 60), int(close_h * 60)
            if open_m >= day_start and close_m <= day_end + 60:
                meal_nodes.append(
                    {
                        "id": mid,
                        "name": name,
                        "category": "meal",
                        "duration_minutes": dur,
                        "preference_score": 0.0,
                        "_open": open_m,
                        "_close": close_m - dur,
                        "_meal": True,
                    }
                )

    # index 0 = depot; then activities; then meals
    nodes = [
        {"id": "_depot", "duration_minutes": 0, "_open": day_start, "_close": day_end, "_depot": True},
        *feasible,
        *meal_nodes,
    ]
    n = len(nodes)
    matrix = _expand_matrix(base, len(meal_nodes))
    for j in range(1, n):
        matrix[0][j] = min(matrix[0][j], _MAX_DEPOT_MINUTES)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return int(matrix[a][b]) + int(nodes[a]["duration_minutes"])

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Skipping an activity costs more the longer it is and the more the traveller wants it; meals are soft.
    for i, node in enumerate(nodes):
        if node.get("_depot"):
            continue
        if node.get("_meal"):
            penalty = 600
        else:
            pref = float(node.get("preference_score") or 0.5)
            penalty = int(400 + 4 * node["duration_minutes"] * (0.5 + pref) + 600 * pref)
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

    # Waiting is allowed for the whole day: a sight that opens at 10:00 in a 07:00 day is a 3h wait.
    horizon = day_end + 90
    routing.AddDimension(transit_idx, horizon - day_start, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for i, node in enumerate(nodes):
        time_dim.CumulVar(manager.NodeToIndex(i)).SetRange(int(node["_open"]), int(node["_close"]))
    time_dim.CumulVar(routing.Start(0)).SetRange(day_start, day_end)
    time_dim.CumulVar(routing.End(0)).SetRange(day_start, day_end + 90)
    # Finish as early as possible (an idle morning is not free), starting at the earliest feasible time.
    time_dim.SetCumulVarSoftUpperBound(routing.End(0), day_start, 1)
    routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(0)))
    routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(0)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = 3  # path cheapest arc
    params.local_search_metaheuristic = 2  # guided local search
    params.time_limit.FromMilliseconds(max(150, int(float(max_time_in_seconds) * 1000)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"scheduled": [], "meals": [], "selected_ids": [], "status": "INFEASIBLE"}

    scheduled: list[dict] = []
    meals: list[dict] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = nodes[manager.IndexToNode(index)]
        if not node.get("_depot"):
            start_m = solution.Value(time_dim.CumulVar(index))
            row = {
                "id": node["id"],
                "name": node["name"],
                "start_hour": round(start_m / 60, 2),
                "end_hour": round((start_m + int(node["duration_minutes"])) / 60, 2),
                "category": node.get("category", "general"),
            }
            (meals if node.get("_meal") else scheduled).append(row)
        index = solution.Value(routing.NextVar(index))

    scheduled.sort(key=lambda x: x["start_hour"])
    return {
        "scheduled": scheduled,
        "meals": meals,
        "selected_ids": [s["id"] for s in scheduled],
        "status": "FEASIBLE" if scheduled else "EMPTY",
    }
