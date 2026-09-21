"""Test fixtures. Live API keys are cleared so runs stay offline."""
from __future__ import annotations

import os

import tempfile

os.environ["ODYSSEY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="odyssey-test-"), "test.db")
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["ODYSSEY_DISABLE_LLM"] = "1"
os.environ["FOURSQUARE_API_KEY"] = ""
os.environ["MAPBOX_API_KEY"] = ""

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


class ScriptedLLM:
    """Stands in for the language model in offline tests: each call to `llm_decide` returns the next
    scripted decision (or calls it with the user prompt), so agent logic is tested without a network."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.prompts: list[str] = []

    def __call__(self, llm, tools, system, user, max_rounds=3):
        self.prompts.append(user)
        item = self.decisions.pop(0) if self.decisions else {}
        return item(user) if callable(item) else dict(item)


@pytest.fixture
def scripted_llm(monkeypatch):
    """Route the Edit Router's model calls to a script. Usage: `llm = scripted_llm([{...}, {...}])`."""

    def install(decisions):
        script = ScriptedLLM(decisions)
        monkeypatch.setattr("agents.edit_router.get_llm", lambda: object())
        monkeypatch.setattr("agents.edit_router.llm_decide", script)
        return script

    return install


# The Trip Analyst is LLM-only, so graph-level tests script what the model would say for each request.
SPECS = {
    "I have 5 days in Kerala with 3 friends. We have around 40000 rupees. We like nature and adventure, but relaxed pace.":
        dict(days=5, travellers=4, budget=40000, prefs={"nature": 0.9, "adventure": 0.9}, pace="relaxed"),
    "5 day trip to Kerala for 2 people, budget 50000, love nature and adventure.":
        dict(days=5, travellers=2, budget=50000, prefs={"nature": 0.9, "adventure": 0.8}),
    "5 day trip to Kerala for 2 people, budget 50000, love nature.": dict(days=5, travellers=2, budget=50000, prefs={"nature": 0.9}),
    "4 day trip to Kerala for 2 people, budget 35000, relaxing beach vibes.":
        dict(days=4, travellers=2, budget=35000, prefs={"relaxation": 0.9}),
    "5 day trip to Kerala for 2 people, budget 60000, love nature and adventure, relaxed pace.":
        dict(days=5, travellers=2, budget=60000, prefs={"nature": 0.9, "adventure": 0.9}, pace="relaxed"),
    "5 days in Kerala with 3 friends, around 60000 rupees, nature and adventure at a relaxed pace, from Delhi":
        dict(days=5, travellers=4, budget=60000, origin="Delhi", prefs={"nature": 0.9, "adventure": 0.9}, pace="relaxed"),
    "plan something nice for me": dict(region="", question="Where would you like to go?"),
}


def scripted_spec(raw: str):
    """Stands in for `_llm_extract`, so it returns its (spec, decision) pair. The decision carries the
    tool names a real model would have called, which the timeline shows."""
    from agents.trip_analyst import _finalize
    from models.schemas import PreferenceWeights, TripConstraints, TripSpec

    kw = SPECS.get(raw)
    if kw is None:
        return None, {}
    region = kw.get("region", "Kerala, India")
    return _finalize(
        TripSpec(
            destination_region=region,
            clarifying_question=kw.get("question"),
            origin_city=kw.get("origin"),
            duration_days=kw.get("days", 5),
            travellers=kw.get("travellers", 1),
            budget_inr=kw.get("budget", 40000),
            preferences=PreferenceWeights(**kw.get("prefs", {})),
            constraints=TripConstraints(pace=kw.get("pace", "moderate")),
            raw_input=raw,
        )
    ), {"_tool_calls": ["geocode_location"]}


@pytest.fixture
def use_scripted_analyst(monkeypatch):
    monkeypatch.setattr("agents.trip_analyst._llm_extract", scripted_spec)


# ---------------------------------------------------------------------------------------------------------------------
# The fake world. The product has no built-in data: places, sights, hotels and prices come from the LLM, and
# coordinates from a geocoder. Offline tests serve a small Kerala world (tests/data/kerala_world.json) through those
# same code paths, so they exercise the real path rather than a shortcut.
# ---------------------------------------------------------------------------------------------------------------------

import json as _json
import re as _re
from pathlib import Path as _Path

WORLD = _json.loads((_Path(__file__).parent / "data" / "kerala_world.json").read_text())
_EXTRA_PLACES = {  # well-known cities the tests name as origins
    "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777), "chennai": (13.0827, 80.2707),
    "bangalore": (12.9716, 77.5946), "jaipur": (26.9124, 75.7873), "goa": (15.2993, 74.1240), "lisbon": (38.7223, -9.1393),
}
_COORDS = {d["name"].lower(): (d["coordinates"]["lat"], d["coordinates"]["lng"]) for d in WORLD["destinations"]} | _EXTRA_PLACES


def fake_nominatim(place_name: str):
    key = place_name.split(",")[0].strip().lower()
    if key in _COORDS:
        lat, lng = _COORDS[key]
        return {"name": place_name.split(",")[0].strip(), "lat": lat, "lng": lng, "source": "osm"}
    return None


def fake_osrm(o: dict, d: dict):
    from tools.mapbox_api import haversine_km

    km = haversine_km(o["lat"], o["lng"], d["lat"], d["lng"]) * 1.4
    return {"distance_km": round(km, 1), "duration_hours": round(km / 40, 2), "cost_inr": round(km * 7), "mode": "road", "source": "osrm"}


def fake_llm_json(system: str, user: str) -> dict:
    s = system.lower()
    if "travel-destination expert" in s:
        if not any(name in user.lower() for name in ("kerala", *(d["name"].lower() for d in WORLD["destinations"]))):
            return {}
        return {"places": [{"name": d["name"], "lat": d["coordinates"]["lat"], "lng": d["coordinates"]["lng"], "description": d["description"],
                            "tags": d["tags"], "activity_scores": d["activity_scores"]} for d in WORLD["destinations"]]}
    if "list real hotels" in s:
        city = (_re.search(r"hotels in (.+?) for a", user) or [None, ""])[1]
        return {"hotels": WORLD["hotels"].get(city, [])}
    if "food costs" in s:
        tier = (_re.search(r"a (\w+)-tier", user) or [None, "mid"])[1]
        return {"per_person_per_day_inr": WORLD["food_cost_per_person_per_day_inr"].get(tier, 900)}
    if "travel quotes" in s:
        return {"available": False}
    return {}


def fake_llm_decide(llm, tools, system, user, max_rounds=3):
    if "visitor attractions" in system:
        city = (_re.search(r"City: (.+?)\.", user) or [None, ""])[1]
        lat, lng = _COORDS.get(city.lower(), (None, None))
        return {"attractions": [{"name": a["name"], "category": a["category"], "duration_minutes": a["duration_minutes"], "rating": a["rating"],
                                 "opening_hour": a["opening_hour"], "closing_hour": a["closing_hour"], "cost_inr": a["cost_inr"], "lat": lat, "lng": lng}
                                for a in WORLD["activities"].get(city, [])]}
    return {}


def install_fake_world(patch):
    """Point every data source at the fake world. `patch` is a MonkeyPatch (function or module scoped)."""
    patch.setattr("tools.mapbox_api._nominatim", fake_nominatim)
    patch.setattr("tools.mapbox_api._osrm", fake_osrm)
    for module in ("tools.foursquare_api", "tools.travel_market", "tools.cost_calculator", "agents.budget_agent"):
        patch.setattr(f"{module}.llm_json", fake_llm_json)
    patch.setattr("tools.foursquare_api.get_llm", lambda: object())
    patch.setattr("tools.foursquare_api.llm_decide", fake_llm_decide)
    patch.setattr("tools.weather._fetch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    for cache in ("tools.foursquare_api._destination_cache", "tools.foursquare_api._attraction_cache", "tools.travel_market._hotel_cache",
                  "tools.travel_market._transport_cache", "tools.cost_calculator._food_cache", "tools.mapbox_api._geocode_cache",
                  "tools.mapbox_api._directions_cache"):
        module, name = cache.rsplit(".", 1)
        getattr(__import__(module, fromlist=[name]), name).clear()


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    """Offline tests never touch a network or a model: the fake world answers instead."""
    install_fake_world(monkeypatch)


# ---------------------------------------------------------------------------------------------------------------------
# Invariants every finished plan must satisfy, whatever produced it — a first plan, an edit, or a disruption.
# A test that only checks the one thing it asked for would not notice the change breaking something else,
# so every path that returns a plan is run through this.
# ---------------------------------------------------------------------------------------------------------------------

def assert_plan_is_coherent(done: dict, *, why: str = "") -> None:
    """Check a serialized plan (the `done` SSE payload or GET /api/itinerary) against itself.

    These are properties, not expected values: they hold for any region, length, party size or wording.
    """
    import math
    from datetime import date as _date, timedelta as _timedelta

    where = f" [{why}]" if why else ""
    itinerary, budget, route = done.get("itinerary"), done.get("budget"), done.get("route")
    assert itinerary, f"a finished plan has an itinerary{where}"
    days = itinerary["days"]
    trip = done.get("trip") or {}

    # Days are numbered 1..n, dated consecutively, and match the length the traveller asked for.
    assert [d["day_number"] for d in days] == list(range(1, len(days) + 1)), f"day numbering{where}"
    if trip.get("duration_days"):
        assert len(days) == trip["duration_days"], f"{len(days)} days for a {trip['duration_days']}-day trip{where}"
    if days[0].get("date"):
        start = _date.fromisoformat(days[0]["date"])
        assert [d["date"] for d in days] == [(start + _timedelta(days=i)).isoformat() for i in range(len(days))], f"dates{where}"

    # Nothing is scheduled twice, a sightseeing day actually has a sight, and the last night is not booked.
    for day in days:
        items = sorted(day["items"], key=lambda i: i["start_hour"])
        assert all(0 <= i["start_hour"] < i["end_hour"] <= 24 for i in items), f"day {day['day_number']} hours{where}"
        assert all(a["end_hour"] <= b["start_hour"] + 1e-6 for a, b in zip(items, items[1:])), f"day {day['day_number']} overlaps{where}"
        if day["kind"] == "sightseeing":
            assert any(i["kind"] == "activity" for i in items), f"day {day['day_number']} is a sightseeing day with no sight{where}"
    assert days[-1].get("overnight_hotel") is None, f"the last day has no overnight stay{where}"

    # A missing hotel is allowed, but never silently: it has to be reported.
    issues = {i["type"] for i in done.get("issues") or []}
    if not all(d.get("overnight_hotel") for d in days[:-1]):
        assert issues & {"NO_HOTEL", "NO_BASE"}, f"a night without a hotel was not reported{where}"

    # Every city in the schedule is one the route actually goes to, so nothing is scheduled unreachably...
    if route and route.get("ordered_destinations"):
        assert {d["destination"] for d in days} <= set(route["ordered_destinations"]), f"a day in a city not on the route{where}"
        # ...and nothing is routed to, and paid for, that the schedule never visits.
        assert set(route["ordered_destinations"]) <= {d["destination"] for d in days}, f"the route pays for a city no day visits{where}"

    if route:
        legs = list(route.get("legs") or []) + ([route["return_leg"]] if route.get("return_leg") else [])
        assert route["total_cost_inr"] == pytest.approx(sum(l["cost_inr"] for l in legs)), f"route total{where}"
        for leg in legs:  # a leg nobody could take must say so, not read as free and instant
            if leg.get("source") == "unavailable":
                assert issues, f"an unroutable leg was not reported{where}"

    # The bill adds up, and the sights billed are exactly the sights scheduled.
    if budget:
        parts = budget["hotels_inr"] + budget["food_inr"] + budget["activities_inr"] + budget["transport_inr"] + budget.get("misc_inr", 0)
        assert budget["total_inr"] == pytest.approx(parts), f"budget parts do not sum to the total{where}"
        assert budget["total_inr"] == pytest.approx(sum(li["amount_inr"] for li in budget["line_items"])), f"line items{where}"
        travellers = trip.get("travellers") or 1
        scheduled = sum(i["cost_inr"] for d in days for i in d["items"] if i["kind"] == "activity") * travellers
        assert budget["activities_inr"] == pytest.approx(scheduled), f"billed sights are not the scheduled sights{where}"
        nights = sum(1 for d in days[:-1] if d.get("overnight_hotel"))
        rooms = math.ceil(travellers / 2)
        priced = sum(d["overnight_hotel"]["price_per_night_inr"] for d in days[:-1] if d.get("overnight_hotel")) * rooms
        assert budget["hotels_inr"] == pytest.approx(priced), f"{nights} night(s) billed against the wrong hotels{where}"
        if budget["total_inr"] > budget["ceiling_inr"] + 1e-6:
            assert "BUDGET" in issues, f"over budget and not reported{where}"

    # Nothing is ever wrong-but-quiet.
    assert done.get("valid") or done.get("issues"), f"invalid with no issue to explain it{where}"
