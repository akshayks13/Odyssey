"""The design document says what the code says."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.strategies import STRATEGIES
from core.team import Team

PLAN = (Path(__file__).resolve().parents[2] / "PLAN.md").read_text()


def test_this_design_document_lists_every_agents_peas():
    for agent in Team().agents.values():
        assert agent.label in PLAN and agent.role in PLAN and agent.architecture in PLAN, agent.name
        for aspect, text in agent.peas.items():
            assert text in PLAN, f"{agent.name}: the {aspect} row of its PEAS is missing or out of date in PLAN.md"
        assert all(f"`{key}`" in PLAN for key in agent.reads + agent.writes)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_is_described(name):
    assert f"`{name}`" in PLAN
