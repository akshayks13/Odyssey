"""Shared fixtures for Odyssey tests. All tests run without live API keys
so they stay deterministic and offline-friendly."""
from __future__ import annotations

import os

os.environ["ODYSSEY_DISABLE_LLM"] = "1"
# Keep pytest offline even when backend/.env has live keys.
os.environ["OPENWEATHER_API_KEY"] = ""
os.environ["FOURSQUARE_API_KEY"] = ""
os.environ["MAPBOX_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest

from models.schemas import (
    Activity,
    Coordinates,
    Destination,
    PreferenceWeights,
    TripConstraints,
    TripSpec,
)
from orchestration.state import initial_state


@pytest.fixture
def trip_spec() -> TripSpec:
    return TripSpec(
        destination_region="Kerala, India",
        duration_days=5,
        travellers=2,
        budget_inr=50000,
        preferences=PreferenceWeights(nature=0.9, adventure=0.8, relaxation=0.6),
        constraints=TripConstraints(max_destinations=3, max_daily_travel_hours=4.0),
        raw_input="5 day trip to Kerala for 2 people, budget 50000, love nature and adventure.",
    )


@pytest.fixture
def sample_destinations() -> list[Destination]:
    return [
        Destination(
            name="Munnar",
            region="Kerala, India",
            coordinates=Coordinates(lat=10.0889, lng=77.0595),
            preference_score=0.91,
            description="Hill station",
            tags=["nature", "adventure"],
        ),
        Destination(
            name="Thekkady",
            region="Kerala, India",
            coordinates=Coordinates(lat=9.5916, lng=77.1667),
            preference_score=0.84,
            description="Wildlife sanctuary",
            tags=["nature", "wildlife"],
        ),
        Destination(
            name="Alleppey",
            region="Kerala, India",
            coordinates=Coordinates(lat=9.4981, lng=76.3388),
            preference_score=0.70,
            description="Backwaters",
            tags=["relaxation"],
        ),
    ]


@pytest.fixture
def sample_activities() -> dict[str, list[Activity]]:
    return {
        "Munnar": [
            Activity(
                id="munnar_tea",
                name="Tea Plantation Tour",
                destination="Munnar",
                category="nature",
                duration_minutes=150,
                cost_inr=300,
                rating=4.7,
                opening_hour=9,
                closing_hour=17,
                preference_score=0.9,
            ),
            Activity(
                id="munnar_trek",
                name="Trekking Trail",
                destination="Munnar",
                category="adventure",
                duration_minutes=240,
                cost_inr=800,
                rating=4.5,
                opening_hour=7,
                closing_hour=15,
                preference_score=0.85,
            ),
        ]
    }


@pytest.fixture
def planning_state(trip_spec):
    state = initial_state(trip_spec.raw_input)
    state["trip_spec"] = trip_spec
    return state
