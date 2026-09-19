"""Agent-level tests with live APIs mocked out (seed-data path only)."""
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


def test_amadeus_uses_real_iata_not_name_slice():
    from tools.amadeus_api import _city_iata, search_flights, search_hotels

    assert _city_iata("Kochi") == "COK"
    assert _city_iata("Munnar") is None
    flights = search_flights.invoke({"origin": "Kochi", "destination": "Munnar", "date": "2026-10-10"})
    assert flights["available"] is False
    assert flights.get("destination_iata") != "MUN"
    hotels = search_hotels.invoke({"destination": "Munnar", "budget_tier": "mid"})
    assert hotels[0]["source"] in {"seed", "seed_default"}


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
