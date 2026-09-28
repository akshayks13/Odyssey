"""Pydantic models shared by agents and tools."""
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
    max_destinations: int = 6
    pace: str = "moderate"  # relaxed | moderate | packed


class TripSpec(BaseModel):
    """Structured trip specification produced by the Trip Analyst Agent."""

    destination_region: str = Field(default="", description="Broad region/state/country, e.g. 'Goa, India'. Empty when the user did not say.")
    duration_days: int = Field(..., ge=1, le=30)
    travellers: int = Field(default=1, ge=1)
    budget_inr: float = Field(..., gt=0)
    preferences: PreferenceWeights = Field(default_factory=PreferenceWeights)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    start_date: Optional[str] = None  # ISO date, optional
    needs_clarification: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None  # set when the request cannot be planned yet
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


class Destination(BaseModel):
    name: str
    region: str = ""
    coordinates: Coordinates
    preference_score: float = 0.5
    description: str = ""
    weather_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Mobility & Routing outputs
# ---------------------------------------------------------------------------

class TransportMode(str, Enum):
    ROAD = "road"
    RAIL = "rail"


class RouteLeg(BaseModel):
    origin: str
    destination: str
    mode: TransportMode = TransportMode.ROAD
    duration_hours: float = 0.0
    cost_inr: float = 0.0
    summary: Optional[str] = None
    vehicle: Optional[str] = None  # for a road hop: own_car, taxi, tempo_traveller or bus


class Route(BaseModel):
    ordered_destinations: list[str]
    legs: list[RouteLeg] = Field(default_factory=list)
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
    tier: str = ""  # budget | mid | premium


class BudgetLineItem(BaseModel):
    category: str
    amount_inr: float
    notes: str = ""


class BudgetBreakdown(BaseModel):
    hotels_inr: float = 0.0
    food_inr: float = 0.0
    activities_inr: float = 0.0
    transport_inr: float = 0.0
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
    kind: str = "activity"  # activity | meal
    cost_inr: float = 0.0  # per person


class StayBlock(BaseModel):
    """Days and nights spent in one city. Shared by Budget (hotel bill) and Architect (schedule)."""

    destination: str
    days: int
    nights: int
    travel_day: bool = False  # first day is consumed by a long transfer


class DayWeather(BaseModel):
    summary: str = ""
    condition: str = ""
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    rain_mm: float = 0.0
    rain_chance: Optional[int] = None
    rainy: bool = False
    source: str = "monthly average"  # monthly average (the agents' belief) | field report (what really happened)


class ItineraryDay(BaseModel):
    day_number: int
    destination: str
    date: Optional[str] = None  # ISO date
    kind: str = "sightseeing"  # sightseeing | travel | leisure
    note: Optional[str] = None
    weather: Optional[DayWeather] = None
    items: list[ScheduledItem] = Field(default_factory=list)
    travel_leg: Optional[RouteLeg] = None
    overnight_hotel: Optional[Hotel] = None


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
    type: str  # BUDGET | BUDGET_CUT | SCHEDULE_CONFLICT | WEATHER | CLOSURE | TRANSPORT | TRAVEL_OVERLOAD | LEISURE_DAY | ALL_TRAVEL | HEAVY_RAIN | SIGHT_CLOSED
    day: Optional[int] = None
    severity: IssueSeverity = IssueSeverity.MEDIUM
    message: str = ""
    target_agent: Optional[str] = None  # which agent should fix this
    avoid: list[str] = Field(default_factory=list)  # places the fix must not use again


class ValidationReport(BaseModel):
    valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)  # everything found; only issues above LOW make the plan invalid
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
    date: Optional[str] = None  # ISO date, when the event is tied to one day (a field report); None = the whole trip
    new_budget_inr: Optional[float] = None


class ObservedFacts(BaseModel):
    """What the team has actually found out about the world, as opposed to what it expects.

    The agents plan on averages (a month's typical weather). The environment only tells them what is
    real for the days, sights and journeys they ask about, so this grows as plans are checked.
    Keys are "<city>|<date>", "<sight id>|<date>".
    """

    heavy_rain: list[str] = Field(default_factory=list)  # "city|date": boating, treks and viewpoints are off
    dry: list[str] = Field(default_factory=list)  # "city|date": checked and fine (overrides a rainy expectation)
    closed: list[str] = Field(default_factory=list)  # "sight id|date"
    checked: list[str] = Field(default_factory=list)  # every fact already asked about, so nothing is checked twice
    checks: int = 0  # how many facts were asked about in total (the cost of sensing)

    def is_heavy_rain(self, city: str, day: str) -> bool:
        return f"{city}|{day}" in self.heavy_rain

    def is_dry(self, city: str, day: str) -> bool:
        return f"{city}|{day}" in self.dry

    def is_closed(self, sight_id: str, day: str) -> bool:
        return f"{sight_id}|{day}" in self.closed


# ---------------------------------------------------------------------------
# Prompt-based editing
# ---------------------------------------------------------------------------

class EditLocks(BaseModel):
    """What the user has asked for so far. Every specialist reads these, so a later replan does not undo an edit."""

    pinned_cities: list[str] = Field(default_factory=list)
    excluded_cities: list[str] = Field(default_factory=list)
    preferred_mode: Optional[str] = None  # road | rail | air
    hotel_prefs: dict[str, str] = Field(default_factory=dict)  # city (lowercase) or "*" -> cheapest | best | hotel name
    excluded_activities: list[str] = Field(default_factory=list)
    pinned_activities: dict[str, int] = Field(default_factory=dict)  # activity name -> day number
    free_days: list[int] = Field(default_factory=list)
    light_days: list[int] = Field(default_factory=list)
    pace: Optional[str] = None  # relaxed | moderate | packed
    day_start_hour: Optional[float] = None


class EditDirective(BaseModel):
    """What the Edit Router decided a user message means."""

    intent: str = "modify"  # modify | answer
    summary: str = ""
    reply: Optional[str] = None  # the answer, for a question
    spec_patch: dict = Field(default_factory=dict)  # TripSpec fields to change
    add_cities: list[str] = Field(default_factory=list)
    remove_cities: list[str] = Field(default_factory=list)
    lock_updates: EditLocks = Field(default_factory=EditLocks)
    disruptions: list[Disruption] = Field(default_factory=list)
    entry: str = "answer"  # agent to re-run from, or "answer"
