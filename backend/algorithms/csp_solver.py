"""OR-Tools VRPTW scheduler for a single day."""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp

_LUNCH = ("_meal_lunch", 12.0, 14.0, 45)
_DINNER = ("_meal_dinner", 19.0, 21.0, 45)


def _default_matrix(n: int, between_minutes: int = 12) -> list[list[int]]:
    return [[0 if i == j else between_minutes for j in range(n)] for i in range(n)]


def _expand_matrix(base: list[list[int]], extra: int, fill: int = 12) -> list[list[int]]:
    if extra <= 0:
        return [row[:] for row in base]
    n = len(base) + extra
    matrix = [row[:] + [fill] * extra for row in base]
    for _ in range(extra):
        matrix.append([fill] * n)
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
        day_start_hour: depot / day start.
        day_end_hour: latest return to depot.
        travel_matrix_minutes: square matrix over [depot] + activities.
            If omitted, a small intra-city default is used.
        include_meals: add lunch/dinner dummy nodes with time windows.

    Returns:
        scheduled (real activities only), selected_ids, status.
    """
    if not activities:
        return {"scheduled": [], "selected_ids": [], "status": "NO_ACTIVITIES"}

    day_start = int(day_start_hour * 60)
    day_end = int(day_end_hour * 60)

    feasible: list[dict] = []
    for act in activities:
        dur = max(15, int(act["duration_minutes"]))
        open_m = max(day_start, int(act["opening_hour"]) * 60)
        close_m = min(day_end, int(act["closing_hour"]) * 60)
        if close_m - open_m < dur:
            continue
        feasible.append({**act, "duration_minutes": dur, "_open": open_m, "_close": close_m - dur})

    if not feasible:
        return {"scheduled": [], "selected_ids": [], "status": "INFEASIBLE"}

    meal_nodes: list[dict] = []
    if include_meals:
        for mid, open_h, close_h, dur in (_LUNCH, _DINNER):
            open_m, close_m = int(open_h * 60), int(close_h * 60)
            if close_m - open_m >= dur and open_m >= day_start and close_m <= day_end + 60:
                meal_nodes.append(
                    {
                        "id": mid,
                        "name": mid,
                        "category": "meal",
                        "duration_minutes": dur,
                        "preference_score": 0.0,
                        "_open": open_m,
                        "_close": max(open_m, close_m - dur),
                        "_meal": True,
                    }
                )

    # index 0 = depot; 1..F = activities; then meals
    nodes = [
        {"id": "_depot", "duration_minutes": 0, "_open": day_start, "_close": day_end, "_depot": True},
        *feasible,
        *meal_nodes,
    ]
    n = len(nodes)

    act_count = len(feasible)
    if travel_matrix_minutes and len(travel_matrix_minutes) == act_count + 1:
        matrix = _expand_matrix(travel_matrix_minutes, len(meal_nodes))
    else:
        matrix = _default_matrix(n)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        service = int(nodes[a]["duration_minutes"])
        return int(matrix[a][b]) + service

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Prefer high-preference activities by making unused-node penalties scale with score.
    for i, node in enumerate(nodes):
        if node.get("_depot"):
            continue
        index = manager.NodeToIndex(i)
        if node.get("_meal"):
            routing.AddDisjunction([index], 250)  # soft meal
        else:
            penalty = int(400 + 800 * float(node.get("preference_score") or 0.5))
            routing.AddDisjunction([index], penalty)

    # Cumul = minutes from midnight. Capacity must cover the latest closing hour.
    routing.AddDimension(
        transit_idx,
        90,  # allow waiting (slack) for time windows
        day_end + 90,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetSpanCostCoefficientForAllVehicles(1)

    for i, node in enumerate(nodes):
        index = manager.NodeToIndex(i)
        time_dim.CumulVar(index).SetRange(int(node["_open"]), int(node["_close"]))

    for v in range(routing.vehicles()):
        time_dim.CumulVar(routing.Start(v)).SetRange(day_start, day_start + 60)
        time_dim.CumulVar(routing.End(v)).SetRange(day_start, day_end)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = 3  # path cheapest arc
    params.local_search_metaheuristic = 2  # guided local search
    params.time_limit.FromMilliseconds(max(150, int(float(max_time_in_seconds) * 1000)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"scheduled": [], "selected_ids": [], "status": "INFEASIBLE"}

    scheduled: list[dict] = []
    selected_ids: list[str] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_i = manager.IndexToNode(index)
        node = nodes[node_i]
        if not node.get("_depot") and not node.get("_meal"):
            start_m = solution.Value(time_dim.CumulVar(index))
            end_m = start_m + int(node["duration_minutes"])
            scheduled.append(
                {
                    "id": node["id"],
                    "name": node["name"],
                    "start_hour": round(start_m / 60, 2),
                    "end_hour": round(end_m / 60, 2),
                    "category": node.get("category", "general"),
                }
            )
            selected_ids.append(node["id"])
        index = solution.Value(routing.NextVar(index))

    scheduled.sort(key=lambda x: x["start_hour"])
    return {"scheduled": scheduled, "selected_ids": selected_ids, "status": "FEASIBLE"}
