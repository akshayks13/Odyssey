"""The strategies the experiments compare. Each one switches a single design choice off.

    full              A* visiting order, CSP timetable, targeted replanning, field checks   (the system)
    greedy_order      nearest-neighbour order instead of A*; the rest as full                (ablates the search)
    greedy_schedule   first-fit timetable instead of the CSP; the rest as full               (ablates the CSP)
    restart           on any issue re-run every agent from the Destination agent            (ablates targeted replanning)
    static            plan once, never check the field, never replan                        (independent agents, no Critic loop)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    description: str
    order: str = "astar"  # astar | greedy: how Mobility orders the cities
    scheduler: str = "csp"  # csp | greedy: how the Architect fills a day
    replan: str = "targeted"  # targeted | restart | none: what the Critic does with an issue
    observe: bool = True  # whether the plan is checked against the field


STRATEGIES: dict[str, Strategy] = {
    s.name: s
    for s in (
        Strategy("full", "Full system", "A* order, CSP timetable, the Critic sends each issue to the one agent that owns it, field checks"),
        Strategy("greedy_order", "Greedy order (no A*)", "Nearest-neighbour city order instead of A*; everything else as the full system", order="greedy"),
        Strategy("greedy_schedule", "Greedy timetable (no CSP)", "First-fit timetable instead of the CSP; everything else as the full system", scheduler="greedy"),
        Strategy("restart", "Restart on any issue", "Same algorithms, but every issue re-runs all agents from the Destination agent", replan="restart"),
        Strategy("static", "Plan once, no checks", "Same algorithms, but no field checks and no replanning (independent agents)", replan="none", observe=False),
    )
}
