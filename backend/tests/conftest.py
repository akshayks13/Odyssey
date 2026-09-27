"""Test fixtures. Every test runs offline against backend/data."""
from __future__ import annotations

import pytest

from agents.trip_analyst import parse_request
from orchestration.graph import build_graph
from orchestration.state import initial_state

# A fixed start date keeps weather (monthly averages) and so every plan identical run to run.
NATURE_TRIP = "5 days in Kerala with 3 friends from 2027-01-10, around ₹60,000, nature and adventure at a relaxed pace"


@pytest.fixture
def spec():
    return parse_request(NATURE_TRIP)[0]


@pytest.fixture
def run_plan():
    """Run the whole graph for a request; returns (graph, config, final state values)."""
    def run(text: str, thread_id: str = "t"):
        graph = build_graph()
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke(initial_state(text), config)
        return graph, config, graph.get_state(config).values
    return run
