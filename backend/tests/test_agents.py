"""Agent tests with live APIs disabled."""
from __future__ import annotations

from agents.budget_agent import budget_agent_node
from agents.destination_agent import destination_agent_node
from agents.mobility_agent import mobility_agent_node
from agents.trip_analyst import trip_analyst_node
from models.schemas import TripSpec
from orchestration.routing import route_after_critic
from orchestration.state import initial_state
from models.schemas import ReplanDirective, ValidationReport


def _analyst_says(monkeypatch, reply):
    import agents.trip_analyst as analyst

    monkeypatch.setattr(analyst, "get_llm", lambda: object())
    monkeypatch.setattr(analyst, "llm_decide", lambda *a, **k: reply)


def test_trip_analyst_reads_the_models_structured_reply(monkeypatch):
    _analyst_says(monkeypatch, {
        "destination_region": "Delhi, India", "origin_city": "Chennai", "duration_days": 4, "travellers": 2,
        "budget_inr": 50000, "start_date": "2099-01-05", "preferences": {"food": 0.9, "culture": 0.9},
        "constraints": {"pace": "relaxed"}, "assumed": ["travellers"],
    })
    spec: TripSpec = trip_analyst_node(initial_state("From Chennai to Delhi for 4 days, 50000, food and culture"))["trip_spec"]
    assert (spec.destination_region, spec.origin_city, spec.duration_days, spec.travellers) == ("Delhi, India", "Chennai", 4, 2)
    assert spec.preferences.food == 0.9 and spec.constraints.pace == "relaxed" and spec.start_date == "2099-01-05"
    assert spec.needs_clarification == ["travellers"]


def test_a_null_in_the_models_reply_falls_back_to_the_default_not_to_an_error(monkeypatch):
    """Seen live: Groq sent "max_daily_travel_hours": null, and the whole request was rejected as 'model unavailable'."""
    _analyst_says(monkeypatch, {
        "destination_region": "Kerala, India", "duration_days": 4, "travellers": 2, "budget_inr": 45000, "start_date": None,
        "preferences": {"nature": 0.9, "food": None}, "constraints": {"pace": "relaxed", "max_daily_travel_hours": None, "max_destinations": None},
    })
    spec = trip_analyst_node(initial_state("4 days in Kerala"))["trip_spec"]
    assert spec.destination_region == "Kerala, India" and spec.constraints.pace == "relaxed"
    assert spec.constraints.max_daily_travel_hours == 4.0 and spec.preferences.nature == 0.9


def test_trip_analyst_asks_where_instead_of_guessing(monkeypatch):
    _analyst_says(monkeypatch, {"destination_region": "", "clarifying_question": "Where would you like to go?"})
    out = trip_analyst_node(initial_state("plan something nice for me"))
    assert out["trip_spec"].destination_region == "" and out["assistant_reply"] == "Where would you like to go?"


def test_trip_analyst_stops_when_no_model_is_available(monkeypatch):
    import pytest as _pytest

    _analyst_says(monkeypatch, {})  # rate-limited: the model returned nothing
    with _pytest.raises(RuntimeError, match="language model is unavailable"):
        trip_analyst_node(initial_state("2-day Ooty trip as a couple"))


def test_trip_analyst_never_invents_a_destination_from_a_bad_reply(monkeypatch):
    import pytest as _pytest

    _analyst_says(monkeypatch, {"duration_days": 3})  # no destination and no question
    with _pytest.raises(RuntimeError):
        trip_analyst_node(initial_state("something"))


def test_assumptions_say_what_was_filled_in():
    from api.sse import _assumptions

    spec = TripSpec(destination_region="Goa, India", duration_days=4, travellers=2, budget_inr=40000, start_date="2099-01-05",
                    needs_clarification=["budget", "start_date", "made up"])
    assert _assumptions(spec) == ["budget: ₹40,000", "start date: 2099-01-05"]


def test_destination_agent_ranks_and_selects(planning_state):
    out = destination_agent_node(planning_state)
    assert len(out["candidate_destinations"]) >= 2
    selected = out["selected_destinations"]
    assert 1 <= len(selected) <= planning_state["trip_spec"].constraints.max_destinations
    scores = [d.preference_score for d in selected]
    assert scores == sorted(scores, reverse=True)
    assert all(d.name in out["candidate_activities"] for d in selected)


def test_mobility_agent_orders_destinations(planning_state):
    dest_out = destination_agent_node(planning_state)
    state = {**planning_state, **dest_out}
    out = mobility_agent_node(state)
    route = out["route"]
    assert route.ordered_destinations
    assert route.search_algorithm.startswith("weighted_astar")
    assert len(route.legs) == max(0, len(route.ordered_destinations) - 1)
    assert all(leg.mode.value == "road" for leg in route.legs)  # no model: road quotes only


def test_public_transport_is_unavailable_without_a_model_or_service(monkeypatch):
    import tools.travel_market as market

    market._transport_cache.clear()
    assert market.search_public_transport.invoke({"origin": "A", "destination": "B", "mode": "air"})["available"] is False
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": False, "note": "no service"})
    assert market.search_public_transport.invoke({"origin": "A", "destination": "B", "mode": "rail"})["available"] is False
    market._transport_cache.clear()


def test_destination_keeps_named_cities_on_short_trip(planning_state):
    spec = planning_state["trip_spec"]
    spec.duration_days = 2
    spec.origin_city = "Kochi"
    spec.raw_input = "2 days Munnar and Alleppey from Kochi"
    planning_state["trip_spec"] = spec
    out = destination_agent_node(planning_state)
    names = {d.name for d in out["selected_destinations"]}
    assert "Munnar" in names
    assert "Alleppey" in names
    assert "Kochi" not in names


def _fly_state(trip_spec, travellers=2):
    from models.schemas import Coordinates, Destination

    delhi = Destination(name="Delhi", region="Delhi, India", coordinates=Coordinates(lat=28.6139, lng=77.209), preference_score=0.9)
    trip_spec.origin_city, trip_spec.destination_region, trip_spec.travellers = "Chennai", "Delhi, India", travellers
    trip_spec.raw_input = "From Chennai to Delhi for 4 days, budget 50000."
    state = initial_state(trip_spec.raw_input)
    state["trip_spec"], state["selected_destinations"] = trip_spec, [delhi]
    return state


def test_mobility_uses_the_models_mode_choice_and_prices_the_group(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility
    import tools.travel_market as market

    market._transport_cache.clear()
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Chennai → Delhi": "air", "Delhi->Chennai": "air"}})  # any arrow style
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "IndiGo", "duration_hours": 2.8, "price_inr": 5000, "note": ""})
    route = mobility_agent_node(_fly_state(trip_spec, travellers=3))["route"]
    assert [l.mode.value for l in route.legs] == ["air"]
    assert route.legs[0].cost_inr == 5000 * 3, "flights are priced per seat for the whole group"
    assert route.return_leg is not None and route.return_leg.mode.value == "air"
    assert route.total_cost_inr == route.legs[0].cost_inr + route.return_leg.cost_inr
    market._transport_cache.clear()


def test_a_drive_longer_than_the_daily_limit_becomes_a_flight_even_if_the_model_said_road(trip_spec, monkeypatch):
    """Delhi to Chennai is over 30 hours by road: never recommend it unless the traveller asked for a road trip."""
    import agents.mobility_agent as mobility
    import tools.travel_market as market
    from models.schemas import EditLocks

    market._transport_cache.clear()
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Chennai|Delhi": "road", "Delhi|Chennai": "road"}})
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "IndiGo", "duration_hours": 2.8, "price_inr": 5000, "note": ""})
    route = mobility_agent_node(_fly_state(trip_spec))["route"]
    assert route.legs[0].mode.value == "air" and route.return_leg.mode.value == "air"
    assert route.legs[0].duration_hours == 2.8 and route.legs[0].cost_inr == 5000 * 2

    market._transport_cache.clear()
    state = _fly_state(trip_spec)
    state["edit_locks"] = EditLocks(preferred_mode="road")
    assert mobility_agent_node(state)["route"].legs[0].mode.value == "road", "they asked for a road trip"

    market._transport_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": False})
    assert mobility_agent_node(_fly_state(trip_spec))["route"].legs[0].mode.value == "road", "nothing faster exists: stay on the road"
    market._transport_cache.clear()


def test_a_flight_that_is_barely_faster_does_not_replace_a_long_drive(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility
    import tools.travel_market as market
    from models.schemas import Coordinates, Destination

    def route_with_flight_hours(hours):
        market._transport_cache.clear()
        monkeypatch.setattr(mobility, "get_llm", lambda: object())
        monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Chennai|Bangalore": "road", "Bangalore|Chennai": "road"}})
        monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "IndiGo", "duration_hours": hours, "price_inr": 4000})
        state = initial_state("2 days")
        trip_spec.origin_city, trip_spec.travellers = "Chennai", 2
        bangalore = Destination(name="Bangalore", region="Karnataka, India", coordinates=Coordinates(lat=12.9716, lng=77.5946), preference_score=0.9)
        state["trip_spec"], state["selected_destinations"] = trip_spec, [bangalore]
        return mobility_agent_node(state)["route"].legs[0]

    assert route_with_flight_hours(7.0).mode.value == "road", "7h by air door to door vs about 10h by road is not worth the fare"
    assert route_with_flight_hours(3.0).mode.value == "air"
    market._transport_cache.clear()


def _road_state(trip_spec, travellers, disrupted=False):
    from models.schemas import Coordinates, Destination, Disruption, DisruptionType

    state = initial_state("2 days")
    trip_spec.origin_city, trip_spec.travellers = "Kochi", travellers
    munnar = Destination(name="Munnar", region="Kerala, India", coordinates=Coordinates(lat=10.0889, lng=77.0595), preference_score=0.9)
    state["trip_spec"], state["selected_destinations"] = trip_spec, [munnar]
    if disrupted:
        state["disruptions"] = [Disruption(type=DisruptionType.TRANSPORT, target="Munnar", description="bus strike")]
    return state


def test_the_road_vehicle_the_model_chooses_prices_the_journey(trip_spec, monkeypatch):
    """A group of six in a tempo traveller is one vehicle; the same hop in own cars would be two."""
    import agents.mobility_agent as mobility

    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "road", "Munnar|Kochi": "road"}, "road_vehicle": "tempo_traveller"})
    leg = mobility_agent_node(_road_state(trip_spec, 6))["route"].legs[0]
    assert leg.vehicle == "tempo_traveller" and leg.cost_inr == round(leg.distance_km * 25, 2)
    assert "tempo traveller" in leg.summary

    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "road"}, "road_vehicle": "own_car"})
    car = mobility_agent_node(_road_state(trip_spec, 6))["route"].legs[0]
    assert car.vehicle == "own_car" and car.cost_inr == round(car.distance_km * 8 * 2, 2), "six people need two cars"


def test_with_no_vehicle_named_a_road_trip_is_their_own_car_and_a_fly_in_is_a_taxi(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility
    import tools.travel_market as market

    market._transport_cache.clear()
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "road"}, "road_vehicle": "spaceship"})
    assert mobility_agent_node(_road_state(trip_spec, 2))["route"].legs[0].vehicle == "own_car"

    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "IndiGo", "duration_hours": 1.5, "price_inr": 4000})
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "air", "Munnar|Kochi": "road"}})
    state = _road_state(trip_spec, 2)
    route = mobility_agent_node(state)["route"]
    assert route.legs[0].mode.value == "air" and route.return_leg.vehicle == "taxi", "someone who flew in has no car for the way back"
    market._transport_cache.clear()


def test_a_strike_makes_the_journey_slower_and_dearer(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility

    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "road"}, "road_vehicle": "taxi"})
    normal = mobility_agent_node(_road_state(trip_spec, 2))["route"].legs[0]
    struck = mobility_agent_node(_road_state(trip_spec, 2, disrupted=True))["route"].legs[0]
    assert struck.duration_hours == round(normal.duration_hours * 5, 2) or abs(struck.duration_hours - normal.duration_hours * 5) < 0.05
    assert struck.cost_inr == normal.cost_inr * 3


def test_a_short_drive_stays_on_the_road(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility
    import tools.travel_market as market
    from models.schemas import Coordinates, Destination

    market._transport_cache.clear()
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Kochi|Munnar": "road"}})
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "IndiGo", "duration_hours": 0.5, "price_inr": 9000})
    state = initial_state("2 days")
    trip_spec.origin_city, trip_spec.travellers = "Kochi", 2
    munnar = Destination(name="Munnar", region="Kerala, India", coordinates=Coordinates(lat=10.0889, lng=77.0595), preference_score=0.9)
    state["trip_spec"], state["selected_destinations"] = trip_spec, [munnar]
    route = mobility_agent_node(state)["route"]
    assert route.legs[0].mode.value == "road", "a 3-hour drive is not worth a flight"
    market._transport_cache.clear()


def test_mobility_falls_back_to_road_when_the_model_quote_is_unavailable(trip_spec, monkeypatch):
    import agents.mobility_agent as mobility
    import tools.travel_market as market

    market._transport_cache.clear()
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda *a, **k: {"modes": {"Chennai|Delhi": "air"}})
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": False})
    assert mobility_agent_node(_fly_state(trip_spec))["route"].legs[0].mode.value == "road"
    market._transport_cache.clear()


def test_mobility_stays_inside_selected_destinations(sample_destinations, trip_spec):
    from models.schemas import Coordinates, Destination

    kochi = Destination(
        name="Kochi",
        region="Kerala, India",
        coordinates=Coordinates(lat=9.9312, lng=76.2673),
        preference_score=0.7,
        description="Fort Kochi and backwaters gateway",
        tags=["culture"],
    )
    trip_spec.raw_input = "5 day Kerala trip from Delhi, prefer train, budget 50000."
    state = initial_state(trip_spec.raw_input)
    state["trip_spec"] = trip_spec
    state["selected_destinations"] = [kochi, sample_destinations[0]]  # Kochi, Munnar
    out = mobility_agent_node(state)
    route = out["route"]
    cities = set(route.ordered_destinations)
    assert cities <= {"Kochi", "Munnar"}
    assert all(leg.origin in cities and leg.destination in cities for leg in route.legs)
    assert "Delhi" not in {leg.origin for leg in route.legs} | {leg.destination for leg in route.legs}
    for leg in route.legs:
        if leg.origin == "Munnar" or leg.destination == "Munnar":
            assert leg.mode.value == "road"


def test_mobility_honours_a_preferred_mode(sample_destinations, trip_spec, monkeypatch):
    import tools.travel_market as market
    from models.schemas import Coordinates, Destination, EditLocks

    kochi = Destination(name="Kochi", region="Kerala, India", coordinates=Coordinates(lat=9.9312, lng=76.2673), preference_score=0.7)
    trivandrum = Destination(name="Trivandrum", region="Kerala, India", coordinates=Coordinates(lat=8.5241, lng=76.9366), preference_score=0.6)
    market._transport_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"available": True, "operator": "Indian Railways", "duration_hours": 4.5, "price_inr": 400, "note": ""})
    state = initial_state("Kerala trip by train for 4 days, budget 40000.")
    state["trip_spec"], state["selected_destinations"] = trip_spec, [kochi, trivandrum]
    state["edit_locks"] = EditLocks(preferred_mode="rail")
    out = mobility_agent_node(state)
    assert out["route"].legs and all(leg.mode.value == "rail" for leg in out["route"].legs)
    market._transport_cache.clear()


def test_budget_agent_produces_breakdown(planning_state):
    dest_out = destination_agent_node(planning_state)
    state = {**planning_state, **dest_out}
    mob_out = mobility_agent_node(state)
    state = {**state, **mob_out}
    out = budget_agent_node(state)
    breakdown = out["budget_breakdown"]
    assert breakdown.total_inr > 0
    assert breakdown.ceiling_inr == planning_state["trip_spec"].budget_inr
    assert breakdown.hotels_inr + breakdown.food_inr + breakdown.activities_inr + breakdown.transport_inr == breakdown.total_inr
    local = 500 * -(-planning_state["trip_spec"].travellers // 4) * planning_state["trip_spec"].duration_days
    assert breakdown.transport_inr == state["route"].total_cost_inr + local, "the journeys, plus autos and cabs between the sights"
    assert "local" in next(i for i in breakdown.line_items if i.category == "transport").notes
    assert "candidate_activities" not in out
    assert "accommodation_options" in out


def test_route_after_critic_valid_ends():
    state = {
        "validation_report": ValidationReport(valid=True, issues=[]),
        "iteration_count": 1,
        "replan_directives": [],
    }
    assert route_after_critic(state) == "valid"


def test_route_after_critic_budget_issue_targets_budget_agent():
    state = {
        "validation_report": ValidationReport(valid=False, issues=[]),
        "iteration_count": 1,
        "replan_directives": [ReplanDirective(target_agent="budget_agent", reason="over budget")],
    }
    assert route_after_critic(state) == "replan_budget"


def test_route_after_critic_max_iterations():
    state = {
        "validation_report": ValidationReport(valid=False, issues=[]),
        "iteration_count": 99,
        "replan_directives": [ReplanDirective(target_agent="budget_agent", reason="over")],
    }
    assert route_after_critic(state) == "max_iterations"


def test_parse_json_blob():
    from llm import parse_json_blob

    assert parse_json_blob('{"selected": ["Munnar"]}')["selected"] == ["Munnar"]
    assert parse_json_blob('```json\n{"target_agent": "budget_agent"}\n```')["target_agent"] == "budget_agent"
    assert parse_json_blob("no json here") == {}


def test_search_attractions_comes_from_the_model_and_is_empty_without_one(monkeypatch):
    import tools.foursquare_api as fsq

    acts = fsq.search_attractions.invoke({"destination": "Munnar"})
    assert any("tea" in a["name"].lower() or "eravikulam" in a["name"].lower() for a in acts)
    fsq._attraction_cache.clear()
    monkeypatch.setattr(fsq, "llm_decide", lambda *a, **k: {})
    assert fsq.search_attractions.invoke({"destination": "Munnar"}) == []  # no fallback data


def test_itinerary_attaches_overnight_hotel(planning_state):
    from agents.itinerary_architect import itinerary_architect_node

    dest_out = destination_agent_node(planning_state)
    state = {**planning_state, **dest_out}
    mob_out = mobility_agent_node(state)
    state = {**state, **mob_out}
    bud_out = budget_agent_node(state)
    state = {**state, **bud_out}
    out = itinerary_architect_node(state)
    days = out["draft_itinerary"].days
    assert days
    assert days[0].overnight_hotel is not None
    assert days[0].overnight_hotel.name
    assert days[-1].overnight_hotel is None
    assert bud_out["budget_breakdown"].selected_hotels


def test_hotels_come_from_the_model_and_are_empty_without_one(monkeypatch):
    import tools.travel_market as market
    from tools.travel_market import search_hotels

    market._hotel_cache.clear()
    assert search_hotels.invoke({"destination": "Munnar", "budget_tier": "mid"})[0]["source"] != "seed"
    market._hotel_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda system, user: {})
    assert search_hotels.invoke({"destination": "Munnar", "budget_tier": "mid"}) == []
    monkeypatch.setattr(market, "llm_json", lambda system, user: {"hotels": [{"name": "Hotel Avenida", "area": "Centro", "price_per_night_inr": 5200, "rating": 4.4}]})
    hotels = search_hotels.invoke({"destination": "Lisbon", "budget_tier": "mid"})
    assert hotels[0]["name"].startswith("Hotel Avenida") and hotels[0]["source"] != "seed"
    market._hotel_cache.clear()


def test_plan_and_stream_endpoints(use_scripted_analyst):
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.post("/api/plan", json={"message": "5 days Kerala", "thread_id": "http-plan-1"}).json()["thread_id"] == "http-plan-1"
    assert client.get("/api/plan/does-not-exist/stream").status_code == 404
    thread_id = "http-stream-1"
    assert client.post("/api/plan", json={"message": "5 day trip to Kerala for 2 people, budget 50000, love nature.", "thread_id": thread_id}).status_code == 200
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as res:
        body = "".join(res.iter_text())
    assert '"type": "init"' in body and '"type": "done"' in body


def test_a_clarifying_question_does_not_save_a_null_itinerary(use_scripted_analyst):
    """`sse.py` used to call save_itinerary unconditionally, so a vague first message that only
    produced a clarifying question would write itinerary: null for that thread_id — a landmine
    for any later save race on the same id. It must only persist once there's a real plan."""
    import json

    from fastapi.testclient import TestClient

    from api.main import app
    from services.itinerary_service import load_itinerary, load_state

    client = TestClient(app)
    thread_id = "http-ask-1"
    client.post("/api/plan", json={"message": "plan something nice for me", "thread_id": thread_id})
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as res:
        body = "".join(res.iter_text())
    events = [json.loads(line[5:]) for line in body.split("\n\n") if line.startswith("data:")]
    done = next(e for e in events if e["type"] == "done")
    assert done["itinerary"] is None and done["reply"] == "Where would you like to go?"
    assert load_itinerary(thread_id) is None, "a clarifying question must not persist a null itinerary"
    assert load_state(thread_id) is None, "a clarifying question must not persist unusable editable state"


def test_a_corrupt_saved_state_blob_is_treated_as_no_state_not_a_500():
    """load_state ran the deserializer with no try/except, so a schema change or a truncated
    blob turned /api/revise and /api/disrupt into unhandled 500s. It must degrade to "no saved
    state" (a clean 404 at the route level), like a thread that was never saved at all."""
    from sqlalchemy.orm import Session

    from services.itinerary_service import SavedState, engine, load_state

    thread_id = "corrupt-state-1"
    with Session(engine) as session:
        session.merge(SavedState(thread_id=thread_id, state_type="json", state_blob=b"not a valid serialized state"))
        session.commit()
    assert load_state(thread_id) is None

    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    res = client.post("/api/revise", json={"thread_id": thread_id, "message": "hi"})
    assert res.status_code == 404


def test_save_plan_writes_both_rows_in_one_transaction():
    """save_itinerary and save_state used to be two separate commits; a crash or a --reload
    restart between them left a thread with a good itinerary but no state row -- it renders
    fine but every edit 404s with a misleading "unknown thread" (seen live on saved data)."""
    from services.itinerary_service import load_itinerary, load_state, save_plan

    thread_id = "atomic-save-1"
    assert load_itinerary(thread_id) is None and load_state(thread_id) is None
    save_plan(thread_id, {"itinerary": {"days": []}, "score": 0.5}, {"trip_spec": None})
    assert load_itinerary(thread_id) is not None
    assert load_state(thread_id) is not None


def test_a_named_city_is_always_a_stop_even_if_the_model_picks_others(planning_state, monkeypatch):
    import agents.destination_agent as destination

    spec = planning_state["trip_spec"]
    spec.destination_region, spec.raw_input, spec.duration_days = "Munnar, India", "4 days in Munnar", 4
    monkeypatch.setattr(destination, "get_llm", lambda: object())
    monkeypatch.setattr(destination, "llm_decide", lambda *a, **k: {"selected": ["Thekkady", "Alleppey"]})
    names = {d.name for d in destination.destination_agent_node(planning_state)["selected_destinations"]}
    assert "Munnar" in names


def test_the_hotel_follows_the_tier_not_always_the_cheapest():
    from agents.budget_agent import _pick_hotel
    from models.schemas import EditLocks

    rows = [{"name": n, "price_per_night_inr": p, "rating": r} for n, p, r in (("Hostel", 700, 4.0), ("Inn", 2500, 4.1), ("Suites", 5200, 4.4), ("Palace", 12000, 4.8))]
    pick = lambda tier: _pick_hotel(rows, "X", tier, EditLocks())[0]["name"]
    assert (pick("budget"), pick("premium")) == ("Hostel", "Palace") and pick("mid") in {"Inn", "Suites"}


def test_the_requested_place_is_a_stop_even_when_discovery_names_it_differently(planning_state, monkeypatch):
    """Asked for Delhi; discovery says "New Delhi" and "Old Delhi". The model's Agra + Mathura pick must not replace it."""
    import agents.destination_agent as destination

    monkeypatch.setattr(destination, "search_destinations", type("T", (), {"invoke": staticmethod(lambda a: [
        {"name": n, "lat": la, "lng": lo, "description": "", "tags": [], "activity_scores": {"culture": s}}
        for n, la, lo, s in (("Agra", 27.17, 78.0, 0.9), ("Mathura", 27.49, 77.67, 0.8), ("New Delhi", 28.61, 77.2, 0.7), ("Old Delhi", 28.65, 77.23, 0.6))])}))
    spec = planning_state["trip_spec"]
    spec.destination_region, spec.raw_input, spec.duration_days, spec.origin_city = "Delhi, India", "From Chennai to Delhi for 4 days", 4, "Chennai"
    monkeypatch.setattr(destination, "get_llm", lambda: object())
    monkeypatch.setattr(destination, "llm_decide", lambda *a, **k: {"selected": ["Agra", "Mathura"]})
    names = [d.name for d in destination.destination_agent_node(planning_state)["selected_destinations"]]
    assert "New Delhi" in names and "Old Delhi" not in names


def test_a_city_without_a_hotel_shows_none_not_another_citys(planning_state):
    from agents.itinerary_architect import _hotel_for_destination
    from models.schemas import BudgetBreakdown, Hotel

    budget = BudgetBreakdown(selected_hotels=[Hotel(name="Taj Inn", destination="Agra", price_per_night_inr=3000)])
    assert _hotel_for_destination(budget, "Agra").name == "Taj Inn" and _hotel_for_destination(budget, "Mathura") is None


def test_sights_come_from_the_model_never_from_foursquare(monkeypatch):
    import tools.foursquare_api as fsq

    fsq._attraction_cache.clear()
    monkeypatch.setattr(fsq, "_fsq_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Foursquare must not be a source")))
    monkeypatch.setattr(fsq, "llm_decide", lambda *a, **k: {"attractions": [
        {"name": "Amber Fort", "category": "culture", "duration_minutes": 120, "rating": 4.6, "opening_hour": 9, "closing_hour": 17, "cost_inr": 200, "lat": 26.98, "lng": 75.85}]})
    monkeypatch.setattr(fsq, "get_llm", lambda: object())
    acts = fsq.search_attractions.invoke({"destination": "Jaipur"})
    assert [a["name"] for a in acts] == ["Amber Fort"] and acts[0]["cost_inr"] == 200
    fsq._attraction_cache.clear()


# --- the timeline may only claim an algorithm that actually ran ---------------------------------------------------------

def _upto_budget(planning_state, budget_inr):
    planning_state["trip_spec"].budget_inr = budget_inr
    state = {**planning_state, **destination_agent_node(planning_state)}
    state = {**state, **mobility_agent_node(state)}
    return state


def test_a_plan_that_fits_the_budget_is_not_reported_as_trimmed(planning_state):
    """The greedy trimming only runs when the plan starts over the ceiling. Claiming it otherwise is a false trace."""
    out = budget_agent_node(_upto_budget(planning_state, 1_000_000))
    assert out["budget_breakdown"].is_within_budget
    assert not any("greedy" in a.lower() for a in out["agent_meta"]["algorithms"]), out["agent_meta"]


def test_a_plan_that_starts_over_budget_says_it_was_trimmed(planning_state):
    out = budget_agent_node(_upto_budget(planning_state, 3_000))
    assert any("greedy" in a.lower() for a in out["agent_meta"]["algorithms"]), out["agent_meta"]


def test_the_architect_reports_exactly_the_or_tools_solves_that_happened(planning_state, monkeypatch):
    """The count in the trace is the number of real solver calls, and the solver is named correctly (the routing
    solver, not CP-SAT, which is a different OR-Tools solver this project does not use)."""
    import agents.itinerary_architect as architect

    real, calls = architect.solve_day_schedule, []
    monkeypatch.setattr(architect, "solve_day_schedule", lambda *a, **k: calls.append(1) or real(*a, **k))
    state = _upto_budget(planning_state, 60_000)
    state = {**state, **budget_agent_node(state)}
    out = architect.itinerary_architect_node(state)

    claim = " ".join(out["agent_meta"]["algorithms"])
    assert calls, "a plan with sightseeing days must call the solver"
    assert f"({len(calls)} day" in claim and "OR-Tools routing solver" in claim, claim
    assert "CP-SAT" not in claim and "CP-SAT" not in out["agent_messages"][0]


def test_a_plan_with_no_solver_run_does_not_claim_ortools(planning_state, monkeypatch):
    import agents.itinerary_architect as architect

    monkeypatch.setattr(architect, "solve_day_schedule", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    state = _upto_budget(planning_state, 60_000)
    state = {**state, **budget_agent_node(state)}
    from models.schemas import EditLocks

    state["edit_locks"] = EditLocks(free_days=list(range(1, state["trip_spec"].duration_days + 1)))  # every day is a free day
    out = architect.itinerary_architect_node(state)
    assert not any("OR-Tools" in a for a in out["agent_meta"]["algorithms"]), out["agent_meta"]
    assert "OR-Tools" not in out["agent_messages"][0]
