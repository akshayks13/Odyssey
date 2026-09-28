"""The design document says what the code says."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.strategies import STRATEGIES
from core.team import Team

ROOT = Path(__file__).resolve().parents[2]
DOCS = {"PLAN.md": (ROOT / "PLAN.md").read_text(), "docs/review1_design_report.md": (ROOT / "docs" / "review1_design_report.md").read_text()}
PLAN = DOCS["PLAN.md"]


def test_the_design_documents_list_every_agents_peas():
    for agent in Team().agents.values():
        assert agent.label in PLAN and agent.role in PLAN and agent.architecture in PLAN, agent.name
        assert all(f"`{key}`" in PLAN for key in agent.reads + agent.writes)
        for doc, text in DOCS.items():
            for aspect, peas in agent.peas.items():
                assert peas in text, f"{agent.name}: the {aspect} row of its PEAS is missing or out of date in {doc}"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_is_described(name):
    assert f"`{name}`" in PLAN
