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


def test_trip_analyst_extracts_kerala_fields():
    state = initial_state(
        "I have 5 days in Kerala with 3 friends. We have around 40000 rupees. We like nature and adventure."
    )
    out = trip_analyst_node(state)
    spec: TripSpec = out["trip_spec"]
    assert spec.destination_region.lower().startswith("kerala")
    assert spec.duration_days == 5
    assert spec.travellers == 4  # speaker + 3 friends
    assert spec.budget_inr == 40000
    assert spec.preferences.nature >= 0.8
    assert spec.preferences.adventure >= 0.8
    assert out["agent_messages"]


def test_trip_analyst_defaults_missing_fields():
    state = initial_state("Take me somewhere nice")
    out = trip_analyst_node(state)
    spec: TripSpec = out["trip_spec"]
    assert spec.duration_days == 5
    assert spec.budget_inr == 40000
    assert spec.needs_clarification  # at least one assumed default


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
    assert all(leg.mode.value in {"road", "rail", "air"} for leg in route.legs)
    from tools.travel_market import _city_iata

    for leg in route.legs:
        if not _city_iata(leg.origin) or not _city_iata(leg.destination):
            assert leg.mode.value != "air"


def test_transport_mode_rules():
    from tools.transport import pick_mode, quote_transport

    hill_air = quote_transport.invoke({"origin": "Kochi", "destination": "Munnar", "mode": "air"})
    assert hill_air["available"] is False
    assert pick_mode("Kochi", "Munnar") == "road"

    rail = quote_transport.invoke({"origin": "Kochi", "destination": "Alleppey", "mode": "rail"})
    assert rail["available"] is False  # Alleppey has no airport code — Mapbox road
    assert pick_mode("Kochi", "Alleppey") == "road"
    assert pick_mode("Kochi", "Trivandrum") == "rail"

    air = quote_transport.invoke({"origin": "Delhi", "destination": "Kochi", "mode": "air"})
    assert air["available"] is True
    assert pick_mode("Delhi", "Kochi") == "air"

    chennai_delhi = quote_transport.invoke({"origin": "Chennai", "destination": "Delhi", "mode": "air"})
    assert chennai_delhi["available"] is True
    assert chennai_delhi.get("origin_iata") == "MAA"
    assert chennai_delhi.get("destination_iata") == "DEL"
    assert chennai_delhi.get("summary")
    assert pick_mode("Chennai", "Delhi") == "air"


def test_trip_analyst_extracts_origin_and_region():
    out = trip_analyst_node(initial_state("From Chennai to Delhi for 4 days, budget 50000, food."))
    spec = out["trip_spec"]
    assert spec.origin_city == "Chennai"
    assert spec.destination_region.lower().startswith("delhi")


def test_trip_analyst_from_unknown_town():
    out = trip_analyst_node(initial_state("From Kochi to Thalassery for 3 days, budget 20000."))
    spec = out["trip_spec"]
    assert spec.origin_city == "Kochi"
    assert spec.destination_region.lower().startswith("thalassery")
    assert spec.duration_days == 3


def test_trip_analyst_named_city_is_not_kerala():
    out = trip_analyst_node(initial_state("3-day Ooty trip, budget 25000, nature."))
    spec = out["trip_spec"]
    assert spec.destination_region.lower().startswith("ooty")
    assert spec.duration_days == 3


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


def test_mobility_flies_chennai_to_delhi(trip_spec):
    from models.schemas import Coordinates, Destination

    delhi = Destination(
        name="Delhi",
        region="Delhi, India",
        coordinates=Coordinates(lat=28.6139, lng=77.209),
        preference_score=0.9,
        description="Capital",
        tags=["culture", "food"],
    )
    trip_spec.origin_city = "Chennai"
    trip_spec.destination_region = "Delhi, India"
    trip_spec.raw_input = "From Chennai to Delhi for 4 days, budget 50000."
    state = initial_state(trip_spec.raw_input)
    state["trip_spec"] = trip_spec
    state["selected_destinations"] = [delhi]
    out = mobility_agent_node(state)
    assert len(out["route"].legs) == 1
    leg = out["route"].legs[0]
    assert (leg.origin, leg.destination, leg.mode.value) == ("Chennai", "Delhi", "air")
    assert leg.origin_iata == "MAA"
    assert leg.destination_iata == "DEL"
    assert leg.summary
    assert "A*" in out["agent_messages"][0]


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


def test_mobility_honours_user_train_preference(sample_destinations, trip_spec):
    from models.schemas import Coordinates, Destination

    kochi = Destination(
        name="Kochi",
        region="Kerala, India",
        coordinates=Coordinates(lat=9.9312, lng=76.2673),
        preference_score=0.7,
        description="Coastal city",
        tags=["culture"],
    )
    trivandrum = Destination(
        name="Trivandrum",
        region="Kerala, India",
        coordinates=Coordinates(lat=8.5241, lng=76.9366),
        preference_score=0.6,
        description="Capital city",
        tags=["culture"],
    )
    trip_spec.raw_input = "Kerala trip by train for 4 days, budget 40000."
    state = initial_state(trip_spec.raw_input)
    state["trip_spec"] = trip_spec
    state["selected_destinations"] = [kochi, trivandrum]
    out = mobility_agent_node(state)
    assert out["route"].legs
    assert all(leg.mode.value == "rail" for leg in out["route"].legs)
    assert "user=rail" in out["agent_messages"][0]


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
    assert "candidate_activities" not in out
    assert "accommodation_options" in out


def test_search_destinations_kerala_uses_seed_cities():
    from tools.foursquare_api import search_destinations

    results = search_destinations.invoke({"region": "Kerala, India"})
    names = {r["name"] for r in results}
    assert "Munnar" in names
    assert "Alleppey" in names


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


def test_validate_trip_schema():
    from agents.trip_analyst import validate_trip_schema

    ok = validate_trip_schema.invoke(
        {
            "spec": {
                "destination_region": "Kerala, India",
                "duration_days": 5,
                "travellers": 2,
                "budget_inr": 40000,
            }
        }
    )
    assert ok["valid"] is True
    bad = validate_trip_schema.invoke({"spec": {"destination_region": "Kerala"}})
    assert bad["valid"] is False


def test_search_attractions_uses_seed_when_offline():
    from tools.foursquare_api import search_attractions

    acts = search_attractions.invoke({"destination": "Munnar"})
    names = {a["name"].lower() for a in acts}
    assert acts
    assert any("tea" in n or "eravikulam" in n or "station" in n for n in names)
    assert all(a.get("source") == "seed" for a in acts)


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


def test_travel_market_uses_real_iata_not_name_slice():
    from tools.travel_market import _city_iata, search_flights, search_hotels

    assert _city_iata("Kochi") == "COK"
    assert _city_iata("Munnar") is None
    flights = search_flights.invoke({"origin": "Kochi", "destination": "Munnar", "date": "2026-10-10"})
    assert flights["available"] is False
    assert flights.get("destination_iata") != "MUN"
    hotels = search_hotels.invoke({"destination": "Munnar", "budget_tier": "mid"})
    assert hotels[0]["source"] in {"seed", "gemini", "groq"}
    elsewhere = search_hotels.invoke({"destination": "Lisbon", "budget_tier": "mid"})
    assert elsewhere == [] or elsewhere[0]["price_per_night_inr"] > 0


def test_plan_and_stream_endpoints():
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
