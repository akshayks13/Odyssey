"""
Pydantic schemas shared across all Odyssey agents and tools.

These models flow through the LangGraph `TripState` (see orchestration/state.py).
Keeping them in one module avoids circular imports between agents/tools/algorithms.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Trip Analyst outputs
# ---------------------------------------------------------------------------

class PreferenceWeights(BaseModel):
    """User preference profile, each score in [0, 1]."""

    nature: float = 0.5
    adventure: float = 0.5
    food: float = 0.5
    nightlife: float = 0.2
    relaxation: float = 0.5
    culture: float = 0.5
    shopping: float = 0.2

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class TripConstraints(BaseModel):
    max_daily_travel_hours: float = 4.0
    max_destinations: int = 4
    pace: str = "moderate"  # relaxed | moderate | packed


class TripSpec(BaseModel):
    """Structured trip specification produced by the Trip Analyst Agent."""

    destination_region: str = Field(..., description="Broad region/state/country, e.g. 'Kerala, India'")
    duration_days: int = Field(..., ge=1, le=30)
    travellers: int = Field(default=1, ge=1)
    budget_inr: float = Field(..., gt=0)
    preferences: PreferenceWeights = Field(default_factory=PreferenceWeights)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    start_date: Optional[str] = None  # ISO date, optional
    needs_clarification: list[str] = Field(default_factory=list)
    raw_input: str = ""


# ---------------------------------------------------------------------------
# Destination Discovery outputs
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    lat: float
    lng: float


class Activity(BaseModel):
    id: str
    name: str
    destination: str
    category: str = "general"
    duration_minutes: int = 120
    cost_inr: float = 0.0
    rating: float = 4.0
    opening_hour: int = 9   # 24h clock, hour only for simplicity
    closing_hour: int = 18
    preference_score: float = 0.5
    coordinates: Optional[Coordinates] = None
    is_closed: bool = False  # set true by disruption injection


class Destination(BaseModel):
    name: str
    region: str = ""
    coordinates: Coordinates
    preference_score: float = 0.5
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    weather_summary: Optional[str] = None
    weather_risk: bool = False


# ---------------------------------------------------------------------------
# Mobility & Routing outputs
# ---------------------------------------------------------------------------

class TransportMode(str, Enum):
    ROAD = "road"
    RAIL = "rail"
    AIR = "air"


class RouteLeg(BaseModel):
    origin: str
    destination: str
    mode: TransportMode = TransportMode.ROAD
    distance_km: float = 0.0
    duration_hours: float = 0.0
    cost_inr: float = 0.0
    available: bool = True


class Route(BaseModel):
    ordered_destinations: list[str]
    legs: list[RouteLeg] = Field(default_factory=list)
    total_distance_km: float = 0.0
    total_duration_hours: float = 0.0
    total_cost_inr: float = 0.0
    search_algorithm: str = "weighted_astar"
    nodes_expanded: int = 0


# ---------------------------------------------------------------------------
# Budget Optimization outputs
# ---------------------------------------------------------------------------

class Hotel(BaseModel):
    name: str
    destination: str
    price_per_night_inr: float
    rating: float = 4.0
    source: str = "seed"  # "gemini" | "seed" | "generated"


class BudgetLineItem(BaseModel):
    category: str
    amount_inr: float
    notes: str = ""


class BudgetBreakdown(BaseModel):
    hotels_inr: float = 0.0
    food_inr: float = 0.0
    activities_inr: float = 0.0
    transport_inr: float = 0.0
    misc_inr: float = 0.0
    total_inr: float = 0.0
    ceiling_inr: float = 0.0
    over_budget_by_inr: float = 0.0
    selected_hotels: list[Hotel] = Field(default_factory=list)
    line_items: list[BudgetLineItem] = Field(default_factory=list)
    tradeoff_suggestions: list[str] = Field(default_factory=list)

    @property
    def is_within_budget(self) -> bool:
        return self.total_inr <= self.ceiling_inr


# ---------------------------------------------------------------------------
# Itinerary Architect outputs
# ---------------------------------------------------------------------------

class ScheduledItem(BaseModel):
    activity_id: str
    activity_name: str
    destination: str
    start_hour: float  # decimal hour, e.g. 9.5 = 09:30
    end_hour: float
    category: str = "general"


class ItineraryDay(BaseModel):
    day_number: int
    destination: str
    items: list[ScheduledItem] = Field(default_factory=list)
    travel_leg: Optional[RouteLeg] = None


class Itinerary(BaseModel):
    days: list[ItineraryDay] = Field(default_factory=list)
    total_cost_inr: float = 0.0
    optimization_score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Critic & Replanner outputs
# ---------------------------------------------------------------------------

class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationIssue(BaseModel):
    type: str  # BUDGET | SCHEDULE_CONFLICT | WEATHER | CLOSURE | TRANSPORT | TRAVEL_OVERLOAD
    day: Optional[int] = None
    severity: IssueSeverity = IssueSeverity.MEDIUM
    message: str = ""
    target_agent: Optional[str] = None  # which agent should fix this


class ValidationReport(BaseModel):
    valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    score: float = 0.0


class ReplanDirective(BaseModel):
    target_agent: str
    reason: str
    constraints: dict = Field(default_factory=dict)


class AgentConflict(BaseModel):
    subject: str  # e.g. destination name
    claims: dict[str, str]  # agent_name -> claim/complaint
    resolution: Optional[str] = None


# ---------------------------------------------------------------------------
# Disruption events (dynamic re-planning)
# ---------------------------------------------------------------------------

class DisruptionType(str, Enum):
    WEATHER = "weather"
    CLOSURE = "closure"
    TRANSPORT = "transport"
    BUDGET_CUT = "budget_cut"


class Disruption(BaseModel):
    type: DisruptionType
    target: str  # destination or activity name affected
    description: str
    day: Optional[int] = None
    new_budget_inr: Optional[float] = None
