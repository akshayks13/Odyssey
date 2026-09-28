"""Whole-system scenarios through the HTTP API, the same path the web UI takes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.strategies import STRATEGIES
from tests.conftest import NATURE_TRIP, RAINY_TRIP


def _stream(client, thread_id) -> tuple[list[dict], dict, list[dict]]:
    events = []
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    done = next(e for e in events if e["type"] in ("done", "error"))
    assert done["type"] == "done", done
    return [e for e in events if e["type"] == "step_complete"], done, [e for e in events if e["type"] == "message"]


def _plan(client, text, **extra):
    body = {"message": text, "uncertainty": "off", **extra}
    thread_id = client.post("/api/plan", json=body).json()["thread_id"]
    return (thread_id, *_stream(client, thread_id))


def _cities(done):
    return [d["name"] for d in done["selected_destinations"]]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_normal_plan_runs_all_six_agents_and_is_valid(client):
    _, steps, done, _ = _plan(client, NATURE_TRIP)
    assert [s["agent"] for s in steps] == ["trip_analyst", "destination_agent", "mobility_agent", "budget_agent", "itinerary_architect", "critic_replanner"]
    assert done["valid"] and len(done["itinerary"]["days"]) == 5
    assert done["budget"]["total_inr"] <= done["budget"]["ceiling_inr"]
    assert all(d["kind"] == "sightseeing" for d in done["itinerary"]["days"])
    assert any("A*" in a for s in steps for a in s["meta"]["algorithms"])
    assert any("CSP" in a for s in steps for a in s["meta"]["algorithms"])


def test_the_stream_shows_every_message_the_agents_send(client):
    _, _, done, messages = _plan(client, NATURE_TRIP)
    assert [m["kind"] for m in messages][:2] == ["REQUEST", "SPEC_READY"] and messages[-1]["kind"] == "ACCEPT"
    assert all({"from", "to", "summary"} <= set(m) for m in messages)
    assert [m["kind"] for m in done["messages"]] == [m["kind"] for m in messages]
    assert done["stats"]["agent_runs"] == 6 and done["stats"]["messages"] == len(messages)


def test_same_request_gives_the_same_plan(client):
    _, _, a, _ = _plan(client, NATURE_TRIP)
    _, _, b, _ = _plan(client, NATURE_TRIP)
    assert a["itinerary"] == b["itinerary"] and a["budget"] == b["budget"] and a["route"] == b["route"]


def test_same_request_meets_the_same_field_even_when_the_field_is_on(client):
    _, _, a, _ = _plan(client, RAINY_TRIP, uncertainty="normal")
    _, _, b, _ = _plan(client, RAINY_TRIP, uncertainty="normal")
    assert a["itinerary"] == b["itinerary"] and a["field_checks"] == b["field_checks"] > 0


def test_unknown_region_asks_instead_of_guessing(client):
    _, steps, done, messages = _plan(client, "4 days in Delhi for 2, ₹50,000")
    assert [s["agent"] for s in steps] == ["trip_analyst"]
    assert done["itinerary"] is None and "Kerala" in done["reply"]
    assert messages[-1]["kind"] == "NEED_INFO" and messages[-1]["to"] == "traveller"


def test_closing_a_city_replans_from_destination_only(client):
    thread_id, _, before, _ = _plan(client, NATURE_TRIP)
    closed = _cities(before)[0]
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "closure", "target": closed, "description": "Road closed"})
    steps, after, _ = _stream(client, thread_id)
    assert "trip_analyst" not in [s["agent"] for s in steps]
    assert steps[0]["agent"] == "critic_replanner" and steps[1]["agent"] == "destination_agent"
    assert closed not in _cities(after)
    assert len(set(_cities(after)) & set(_cities(before))) >= 1  # the other cities stay where possible


def test_budget_cut_reprices_from_budget(client):
    thread_id, _, _, _ = _plan(client, NATURE_TRIP)
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "budget_cut", "target": "trip", "description": "Cut", "new_budget_inr": 40000})
    steps, after, _ = _stream(client, thread_id)
    assert [s["agent"] for s in steps][:2] == ["critic_replanner", "budget_agent"]
    assert after["budget"]["ceiling_inr"] == 40000


def test_strike_replans_the_route(client):
    thread_id, _, before, _ = _plan(client, NATURE_TRIP)
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "transport", "target": _cities(before)[0], "description": "Strike"})
    steps, _, _ = _stream(client, thread_id)
    assert [s["agent"] for s in steps][:2] == ["critic_replanner", "mobility_agent"]


@pytest.mark.parametrize(
    "message, entry",
    [
        ("make day 2 lighter", "itinerary_architect"),
        ("cheapest hotels", "budget_agent"),
        ("by train", "mobility_agent"),
        ("add Alleppey", "destination_agent"),
    ],
)
def test_edits_rerun_only_from_the_earliest_affected_agent(client, message, entry):
    thread_id, _, _, _ = _plan(client, NATURE_TRIP)
    assert client.post("/api/revise", json={"thread_id": thread_id, "message": message}).status_code == 200
    steps, done, _ = _stream(client, thread_id)
    order = [s["agent"] for s in steps]
    assert order[0] == "edit_router" and order[1] == entry
    assert done["reran_from"] == entry


def test_a_lighter_day_has_one_sight(client):
    thread_id, _, _, _ = _plan(client, NATURE_TRIP)
    client.post("/api/revise", json={"thread_id": thread_id, "message": "make day 3 lighter"})
    _, done, _ = _stream(client, thread_id)
    day3 = done["itinerary"]["days"][2]
    assert sum(1 for i in day3["items"] if i["kind"] == "activity") == 1


def test_a_question_leaves_the_plan_alone(client):
    thread_id, _, before, _ = _plan(client, NATURE_TRIP)
    client.post("/api/revise", json={"thread_id": thread_id, "message": "which hotels?"})
    steps, done, _ = _stream(client, thread_id)
    assert [s["agent"] for s in steps] == ["edit_router"]
    assert "Hotels:" in done["reply"]
    assert client.get(f"/api/itinerary/{thread_id}").json()["itinerary"] == before["itinerary"]


def test_editing_a_thread_that_only_asked_a_question_is_harmless(client):
    thread_id, _, _, _ = _plan(client, "4 days in Delhi for 2")
    assert client.post("/api/revise", json={"thread_id": thread_id, "message": "add Kochi"}).status_code == 200
    _, done, _ = _stream(client, thread_id)
    assert done["itinerary"] is None and "no plan yet" in done["reply"]


def _rain_replans(client, seed) -> bool:
    _, _, _, messages = _plan(client, RAINY_TRIP, uncertainty="high", seed=seed)
    return any(m["kind"] == "REPLAN" and m["summary"].startswith("HEAVY_RAIN") for m in messages)


def test_the_field_can_change_a_plan_and_the_timeline_shows_the_check(client):
    seed = next(s for s in range(60) if _rain_replans(client, s))
    _, steps, done, messages = _plan(client, RAINY_TRIP, uncertainty="high", seed=seed)
    kinds = [m["kind"] for m in messages]
    assert "FIELD_REPORT" in kinds and "REPLAN" in kinds
    assert "environment" in [s["agent"] for s in steps] and done["valid"]


def test_compare_runs_every_strategy_on_this_threads_request(client):
    thread_id, _, _, _ = _plan(client, RAINY_TRIP, uncertainty="high", seed=3)
    body = client.post("/api/compare", json={"thread_id": thread_id}).json()
    assert set(body["strategies"]) == set(STRATEGIES)
    assert body["strategies"]["full"]["value_ratio"] >= body["strategies"]["static"]["value_ratio"]
    assert body["seed"] == 3 and body["uncertainty"] == "high"


def test_compare_replays_the_events_the_traveller_reported(client):
    thread_id, _, _, _ = _plan(client, NATURE_TRIP)
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "budget_cut", "target": "trip", "description": "cut", "new_budget_inr": 40000})
    _stream(client, thread_id)
    body = client.post("/api/compare", json={"thread_id": thread_id}).json()
    assert body["reported"] == 1 and body["strategies"]["full"]["planned"]


def test_agents_endpoint_lists_the_peas_of_every_agent(client):
    agents = client.get("/api/agents").json()
    assert [a["name"] for a in agents][:6] == ["trip_analyst", "destination_agent", "mobility_agent", "budget_agent", "itinerary_architect", "critic_replanner"]
    assert all(a["peas"]["performance"] and a["reads"] and a["writes"] for a in agents)


def test_bad_requests_are_refused(client):
    assert client.post("/api/plan", json={"message": "x", "strategy": "magic"}).status_code == 422
    assert client.post("/api/plan", json={"message": "x", "uncertainty": "chaos"}).status_code == 422
    assert client.get("/api/itinerary/nope").status_code == 404
    assert client.post("/api/revise", json={"thread_id": "nope", "message": "hi"}).status_code == 404


def test_health_says_there_is_no_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model"].startswith("none") and set(body["strategies"]) == set(STRATEGIES)
