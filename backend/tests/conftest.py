"""Test fixtures. Every test runs offline against backend/data, with a fixed field seed."""
from __future__ import annotations

import pytest

from agents.trip_analyst import parse_request
from core.team import Team

# A fixed start date keeps the weather (monthly averages) and so every plan identical run to run.
NATURE_TRIP = "5 days in Kerala with 3 friends from 2027-01-10, around ₹60,000, nature and adventure at a relaxed pace"
RAINY_TRIP = "5 days in Kerala with 3 friends from 2027-07-10, around ₹60,000, nature and adventure at a relaxed pace"


@pytest.fixture
def spec():
    return parse_request(NATURE_TRIP)[0]


@pytest.fixture
def make_team():
    """A team that has already planned `text`. The field is off by default so a plan is the pure planning result."""
    def make(text: str = NATURE_TRIP, strategy: str = "full", seed: int = 1, uncertainty: str = "off") -> Team:
        team = Team(strategy, seed, uncertainty)
        team.run_to_end(team.plan(text))
        return team
    return make
