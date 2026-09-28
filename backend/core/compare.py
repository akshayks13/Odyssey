"""Run the strategies on the same request, the same field and the same reported events, and score them alike.

Used by the demo's "compare strategies" button and by the experiments (`experiments/`). The reference for
"how much of the trip's value was delivered" is the same system planning in a world where nothing surprises it.
"""
from __future__ import annotations

from time import perf_counter

from core.metrics import evaluate
from core.strategies import STRATEGIES
from core.team import Team
from models.schemas import Disruption


def run_strategy(text: str, strategy: str, seed: int, uncertainty: str, disruptions: list[Disruption] | tuple = ()) -> dict:
    """Plan `text` with one strategy, apply the reported events in order, and score the result against the true field."""
    team = Team(strategy, seed, uncertainty)
    started = perf_counter()
    team.run_to_end(team.plan(text))
    for event in disruptions:
        team.run_to_end(team.disrupt(event))
    result = evaluate(team)
    result["wall_ms"] = round(1000 * (perf_counter() - started), 1)
    return result


def compare_strategies(text: str, seed: int, uncertainty: str, disruptions: list[Disruption] | tuple = ()) -> dict[str, dict]:
    """Every strategy's score on this request, plus `value_ratio`: value delivered / value of the trip the same system would plan with no surprises."""
    ideal = run_strategy(text, "full", seed, "off", disruptions)["value_planned"]
    out = {}
    for name, strategy in STRATEGIES.items():
        result = run_strategy(text, name, seed, uncertainty, disruptions)
        result["label"] = strategy.label
        result["value_ratio"] = round(result["value_delivered"] / ideal, 4) if ideal > 0 else 1.0
        out[name] = result
    return out
