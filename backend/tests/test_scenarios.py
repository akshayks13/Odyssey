"""End-to-end graph tests: happy path, weather closure, and budget cut."""
from __future__ import annotations

import pytest

from models.schemas import Disruption, DisruptionType
from orchestration.graph import get_graph
from orchestration.state import initial_state


GRAPH = get_graph()
pytestmark = pytest.mark.usefixtures("use_scripted_analyst")


def _run(raw_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return GRAPH.invoke(initial_state(raw_input), config=config), config


def test_scenario_normal_kerala_trip():
    result, _ = _run(
        "I have 5 days in Kerala with 3 friends. We have around 40000 rupees. We like nature and adventure, but relaxed pace.",
        "scenario-normal",
    )
    assert result["trip_spec"].destination_region.lower().startswith("kerala")
    assert result["trip_spec"].duration_days == 5
    assert result["selected_destinations"]
    assert result["route"] is not None
    assert result["budget_breakdown"] is not None
    itinerary = result.get("final_itinerary") or result.get("draft_itinerary")
    assert itinerary is not None
    assert len(itinerary.days) == 5
    assert any(day.items for day in itinerary.days), "at least one day must have scheduled activities"
    assert result["validation_report"] is not None
    names = [m.split(":")[0] for m in result["agent_messages"]]
    for agent in ("Trip Analyst", "Destination Agent", "Mobility Agent", "Budget Agent", "Itinerary Architect", "Critic"):
        assert any(agent in n for n in names)


def test_scenario_weather_closure_replans_destination():
    result, config = _run(
        "5 day trip to Kerala for 2 people, budget 50000, love nature and adventure.",
        "scenario-closure",
    )
    original = [d.name for d in result["selected_destinations"]]
    assert original
    closed = original[0]

    disruption = Disruption(
        type=DisruptionType.CLOSURE,
        target=closed,
        description="Heavy rain damage, closed for repairs",
        day=1,
    )
    GRAPH.update_state(
        config,
        {"disruptions": [disruption], "iteration_count": 0},
        as_node="itinerary_architect",
    )
    result2 = GRAPH.invoke(None, config=config)

    new_names = [d.name for d in result2["selected_destinations"]]
    assert closed not in new_names, "closed destination must be swapped out"
    assert result2.get("final_itinerary") is not None
    # Critic must have routed through destination_agent (targeted, not a full restart)
    later_messages = result2["agent_messages"][len(result["agent_messages"]) :]
    joined = " ".join(later_messages)
    assert "destination_agent" in joined or "Destination Agent" in joined
    assert "Trip Analyst" not in joined  # analyst is NOT re-invoked on targeted replan


def test_scenario_budget_cut_triggers_budget_agent():
    result, config = _run(
        "4 day trip to Kerala for 2 people, budget 35000, relaxing beach vibes.",
        "scenario-budget-cut",
    )
    original_total = result["budget_breakdown"].total_inr
    original_ceiling = result["budget_breakdown"].ceiling_inr
    assert original_total <= original_ceiling

    disruption = Disruption(
        type=DisruptionType.BUDGET_CUT,
        target="trip",
        description="Sponsor withdrew, budget cut",
        new_budget_inr=8000,
    )
    GRAPH.update_state(
        config,
        {"disruptions": [disruption], "iteration_count": 0},
        as_node="itinerary_architect",
    )
    result2 = GRAPH.invoke(None, config=config)

    later_messages = result2["agent_messages"][len(result["agent_messages"]) :]
    joined = " ".join(later_messages)
    assert "Budget Agent" in joined or "budget_agent" in joined
    assert result2["budget_breakdown"].ceiling_inr == 8000
    # Either repaired under the new ceiling, or returned a best-effort plan with a warning
    assert result2.get("final_itinerary") is not None or result2.get("draft_itinerary") is not None


# --- the plan is internally consistent --------------------------------------------------------------------------

import math
from datetime import date, timedelta

import pytest

import tools.cost_calculator as cost
import tools.foursquare_api as fsq
import tools.mapbox_api as mapbox
import tools.travel_market as market
from agents.trip_analyst import _finalize, trip_analyst_node
from models.schemas import TripSpec

MSG = "5 days in Kerala with 3 friends, around 60000 rupees, nature and adventure at a relaxed pace, from Delhi"


@pytest.fixture(scope="module")
def planned():
    with pytest.MonkeyPatch.context() as patch:  # a module fixture can't use the function-scoped ones
        from tests.conftest import install_fake_world, scripted_spec

        install_fake_world(patch)
        patch.setattr("agents.trip_analyst._llm_extract", scripted_spec)
        return GRAPH.invoke(initial_state(MSG), config={"configurable": {"thread_id": "consistency-1"}})


def _days(result):
    return (result.get("final_itinerary") or result["draft_itinerary"]).days


def check_plan(result):
    """Properties every plan must have, whatever region or wording produced it."""
    spec, days, budget, stay = result["trip_spec"], _days(result), result["budget_breakdown"], result["stay_plan"]
    assert [d.day_number for d in days] == list(range(1, spec.duration_days + 1))
    start = date.fromisoformat(spec.start_date)
    assert [d.date for d in days] == [(start + timedelta(days=i)).isoformat() for i in range(spec.duration_days)]
    for block in stay:  # the bill and the schedule describe the same trip
        assert sum(1 for d in days if d.destination == block.destination) == block.days
    assert sum(b.nights for b in stay) == spec.duration_days - 1
    assert days[-1].overnight_hotel is None
    if not all(d.overnight_hotel for d in days[:-1]):  # a lookup failed: it must be reported, not silent
        assert any(i.type == "NO_HOTEL" for i in result["validation_report"].issues)
    price = {h.destination: h.price_per_night_inr for h in budget.selected_hotels}
    rooms = math.ceil(spec.travellers / 2)
    assert budget.hotels_inr == pytest.approx(sum(price[b.destination] * b.nights * rooms for b in stay if b.destination in price))
    assert budget.total_inr == pytest.approx(budget.hotels_inr + budget.food_inr + budget.activities_inr + budget.transport_inr)
    scheduled = sum(i.cost_inr for d in days for i in d.items if i.kind == "activity") * spec.travellers
    assert budget.activities_inr == pytest.approx(scheduled)
    for d in days:
        items = sorted(d.items, key=lambda i: i.start_hour)
        assert all(0 <= i.start_hour < i.end_hour <= 24 for i in items)
        assert all(a.end_hour <= b.start_hour + 1e-6 for a, b in zip(items, items[1:])), f"day {d.day_number} overlaps"
        if d.kind == "sightseeing":
            assert any(i.kind == "activity" for i in items), f"day {d.day_number} is empty"
    if spec.origin_city:
        assert result["route"].return_leg is not None
        assert result["route"].total_cost_inr == pytest.approx(sum(l.cost_inr for l in result["route"].legs) + result["route"].return_leg.cost_inr)
    report = result["validation_report"]
    assert report.valid or report.issues, "a problem must be reported, never silent"


def test_the_plan_is_internally_consistent(planned):
    check_plan(planned)


def test_no_sight_is_free_by_accident(planned):
    assert planned["budget_breakdown"].activities_inr > 0


# --- the model, not a lookup table, supplies the data -----------------------------------------------------------

def test_destinations_come_from_the_model_for_any_region(monkeypatch):
    fsq._destination_cache.clear()
    monkeypatch.setattr(fsq, "llm_json", lambda system, user: {"places": [
        {"name": "Florence", "lat": 43.77, "lng": 11.25, "description": "Renaissance city", "activity_scores": {"culture": 0.95}},
        {"name": "Siena", "lat": 43.32, "lng": 11.33, "activity_scores": {"culture": 0.9}},
        {"name": "Florence", "lat": 0, "lng": 0},  # duplicate
    ]})
    rows = fsq.search_destinations.invoke({"region": "Tuscany, Italy"})
    assert [r["name"] for r in rows] == ["Florence", "Siena"] and rows[0]["activity_scores"]["culture"] == 0.95
    fsq._destination_cache.clear()


def test_when_the_model_cannot_list_places_nothing_is_invented(monkeypatch):
    """No seed data and no "the region is the only city" guess: an empty list, and the plan stops."""
    fsq._destination_cache.clear()
    monkeypatch.setattr(fsq, "llm_json", lambda system, user: {})
    assert fsq.search_destinations.invoke({"region": "Goa, India"}) == []
    fsq._destination_cache.clear()


def test_the_plan_stops_with_a_clear_message_when_no_places_come_back(monkeypatch):
    from agents.destination_agent import destination_agent_node
    from llm import MODEL_UNAVAILABLE

    monkeypatch.setattr("agents.destination_agent.search_destinations", type("T", (), {"invoke": staticmethod(lambda a: [])}))
    spec = TripSpec(destination_region="Kerala, India", duration_days=5, budget_inr=40000, raw_input="5 days in Kerala")
    with pytest.raises(RuntimeError, match="language model is unavailable"):
        destination_agent_node({"trip_spec": spec})
    assert "sample trip" in MODEL_UNAVAILABLE


def test_geocoding_has_one_fallback_and_then_says_not_found():
    """Mapbox, then OpenStreetMap, then an honest "not found": never a guessed point in the middle of India."""
    hit = mapbox.geocode_location.invoke({"place_name": "Lisbon, Portugal"})
    assert hit["source"] == "osm" and hit["lat"] == pytest.approx(38.72, abs=0.1)
    miss = mapbox.geocode_location.invoke({"place_name": "Ooty As A Couple, India"})
    assert miss["lat"] is None and miss["source"] == "not_found"


def test_food_prices_are_regional_and_a_silly_estimate_is_ignored(monkeypatch):
    cost._food_cache.clear()
    monkeypatch.setattr(cost, "llm_json", lambda system, user: {"per_person_per_day_inr": 2200})
    out = cost.estimate_food_costs.invoke({"duration_days": 3, "travellers": 2, "tier": "mid", "region": "Lisbon, Portugal"})
    assert out["total_inr"] == 2200 * 3 * 2
    cost._food_cache.clear()
    monkeypatch.setattr(cost, "llm_json", lambda system, user: {"per_person_per_day_inr": 65000})
    assert cost.estimate_food_costs.invoke({"duration_days": 3, "travellers": 2, "tier": "mid", "region": "X"})["per_person_per_day_inr"] < 4000
    cost._food_cache.clear()


def test_a_failed_lookup_is_not_remembered(monkeypatch):
    market._hotel_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda system, user: {})  # rate-limited
    assert market.search_hotels.invoke({"destination": "Lisbon", "budget_tier": "mid"}) == []
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"hotels": [{"name": "Hotel Avenida", "price_per_night_inr": 5200, "rating": 4.4}]})
    assert market.search_hotels.invoke({"destination": "Lisbon", "budget_tier": "mid"})[0]["name"] == "Hotel Avenida"
    market._hotel_cache.clear()


# --- weather drives real decisions ------------------------------------------------------------------------------------

def _weather_rows(*rainy_flags):
    return lambda lat, lng, start, end, historical: [
        {"date": start.isoformat(), "tmin": 22.0, "tmax": 29.0, "rain_mm": 14.0 if r else 0.0, "rain_chance": 90 if r else 10,
         "condition": "Rain" if r else "Clear", "rainy": r}
        for r in rainy_flags[: (end - start).days + 1]
    ]


def _sights_state(planning_state):
    from agents.budget_agent import budget_agent_node
    from agents.destination_agent import destination_agent_node
    from agents.mobility_agent import mobility_agent_node
    from models.schemas import Activity, EditLocks

    state = {**planning_state, **destination_agent_node(planning_state)}
    state["selected_destinations"] = state["selected_destinations"][:1]
    city = state["selected_destinations"][0]
    state["candidate_activities"] = {city.name: [
        Activity(id="a1", name="Trek", destination=city.name, category="adventure", duration_minutes=120, preference_score=0.95, coordinates=city.coordinates),
        Activity(id="a2", name="Museum", destination=city.name, category="culture", duration_minutes=120, preference_score=0.6, coordinates=city.coordinates),
        Activity(id="a3", name="Viewpoint", destination=city.name, category="nature", duration_minutes=120, preference_score=0.9, coordinates=city.coordinates),
        Activity(id="a4", name="Fort", destination=city.name, category="culture", duration_minutes=120, preference_score=0.5, coordinates=city.coordinates),
    ]}
    state["edit_locks"] = EditLocks(light_days=[1, 2])  # one sight a day, so the choice is visible
    state["trip_spec"] = state["trip_spec"].model_copy(update={"start_date": "2099-03-01", "duration_days": 2})
    for node in (mobility_agent_node, budget_agent_node):
        state = {**state, **node(state)}
    return state


def test_a_rainy_day_gets_indoor_sights_and_a_dry_day_the_outdoor_ones(planning_state, monkeypatch):
    from agents.itinerary_architect import itinerary_architect_node

    monkeypatch.setattr("tools.weather._fetch", _weather_rows(True, False))  # day 1 rain, day 2 dry
    state = _sights_state(planning_state)
    days = itinerary_architect_node(state)["draft_itinerary"].days
    first = [i.activity_name for i in days[0].items if i.kind == "activity"]
    second = [i.activity_name for i in days[1].items if i.kind == "activity"]
    assert first == ["Museum"], "rain: the indoor sight comes first, the outdoor favourites wait"
    assert second == ["Trek"], "a dry day gets the traveller's favourite"
    assert days[0].weather.rainy and days[0].weather.rain_chance == 90 and "Rain likely" in (days[0].note or "")
    assert days[1].weather.rainy is False


def test_every_day_carries_its_weather_and_source(planning_state, monkeypatch):
    from agents.itinerary_architect import itinerary_architect_node

    monkeypatch.setattr("tools.weather._fetch", _weather_rows(False, False))
    days = itinerary_architect_node(_sights_state(planning_state))["draft_itinerary"].days
    assert all(d.weather and d.weather.source == "last year" and d.weather.tmax == 29.0 for d in days)  # 2099 is beyond the forecast


def test_no_weather_never_stops_a_plan(planning_state):
    from agents.itinerary_architect import itinerary_architect_node

    days = itinerary_architect_node(_sights_state(planning_state))["draft_itinerary"].days  # conftest: weather unavailable
    assert days and all(d.weather is None for d in days)


def test_the_destination_agent_sees_the_rain_for_the_trip_dates(planning_state, monkeypatch):
    import agents.destination_agent as destination

    monkeypatch.setattr("tools.weather._fetch", _weather_rows(True, True, True, True, True))
    seen = []
    monkeypatch.setattr(destination, "get_llm", lambda: object())
    monkeypatch.setattr(destination, "llm_decide", lambda llm, tools, system, user, **k: seen.append(user) or {})
    out = destination.destination_agent_node(planning_state)
    assert all(c.weather_risk for c in out["candidate_destinations"]) and "rain likely on" in out["candidate_destinations"][0].weather_summary
    assert any("rain likely on" in u for u in seen), "the model is shown each candidate's weather"


# --- asking instead of guessing; dates -----------------------------------------------------------------------

def test_no_destination_means_a_question_not_a_guess():
    out = trip_analyst_node(initial_state("plan something nice for me"))
    assert out["trip_spec"].destination_region == "" and out["assistant_reply"]
    result = GRAPH.invoke(initial_state("plan something nice for me"), config={"configurable": {"thread_id": "vague-1"}})
    assert result["assistant_reply"] and not result.get("draft_itinerary")


def test_a_start_date_in_the_past_is_replaced_by_one_in_the_future():
    """A model that is not told today's date guesses last year."""
    stale = TripSpec(destination_region="Goa, India", duration_days=4, budget_inr=30000, start_date="2023-11-08")
    fixed = _finalize(stale)
    assert fixed.start_date == (date.today() + timedelta(days=14)).isoformat() and "start_date" in fixed.needs_clarification
    kept = _finalize(TripSpec(destination_region="Goa, India", duration_days=4, budget_inr=30000, start_date="2099-01-05"))
    assert kept.start_date == "2099-01-05"


def test_a_model_cannot_disable_the_travel_limit():
    spec = TripSpec(destination_region="Goa, India", duration_days=4, budget_inr=30000)
    spec.constraints = spec.constraints.model_copy(update={"max_daily_travel_hours": 24, "max_destinations": 9})
    fixed = _finalize(spec)
    assert fixed.constraints.max_daily_travel_hours == 8.0 and fixed.constraints.max_destinations == 5
