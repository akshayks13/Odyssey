"""Whole-system scenarios through the HTTP API, the same path the web UI takes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import NATURE_TRIP


def _stream(client, thread_id) -> tuple[list[dict], dict]:
    events = []
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    done = next(e for e in events if e["type"] in ("done", "error"))
    assert done["type"] == "done", done
    return [e for e in events if e["type"] == "step_complete"], done


def _plan(client, text):
    thread_id = client.post("/api/plan", json={"message": text}).json()["thread_id"]
    steps, done = _stream(client, thread_id)
    return thread_id, steps, done


def _cities(done):
    return [d["name"] for d in done["selected_destinations"]]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_normal_plan_runs_all_six_agents_and_is_valid(client):
    _, steps, done = _plan(client, NATURE_TRIP)
    assert [s["agent"] for s in steps] == ["trip_analyst", "destination_agent", "mobility_agent", "budget_agent", "itinerary_architect", "critic_replanner"]
    assert done["valid"] and len(done["itinerary"]["days"]) == 5
    assert done["budget"]["total_inr"] <= done["budget"]["ceiling_inr"]
    assert all(d["kind"] == "sightseeing" for d in done["itinerary"]["days"])
    assert any("A*" in a for s in steps for a in s["meta"]["algorithms"])
    assert any("CSP" in a for s in steps for a in s["meta"]["algorithms"])


def test_same_request_gives_the_same_plan(client):
    _, _, a = _plan(client, NATURE_TRIP)
    _, _, b = _plan(client, NATURE_TRIP)
    assert a["itinerary"] == b["itinerary"] and a["budget"] == b["budget"] and a["route"] == b["route"]


def test_unknown_region_asks_instead_of_guessing(client):
    _, steps, done = _plan(client, "4 days in Delhi for 2, ₹50,000")
    assert [s["agent"] for s in steps] == ["trip_analyst"]
    assert done["itinerary"] is None and "Kerala" in done["reply"]


def test_closing_a_city_replans_from_destination_only(client):
    thread_id, _, before = _plan(client, NATURE_TRIP)
    closed = _cities(before)[0]
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "closure", "target": closed, "description": "Road closed"})
    steps, after = _stream(client, thread_id)
    assert "trip_analyst" not in [s["agent"] for s in steps]
    assert steps[0]["agent"] == "critic_replanner" and steps[1]["agent"] == "destination_agent"
    assert closed not in _cities(after)
    assert len(set(_cities(after)) & set(_cities(before))) >= 1  # the other cities stay where possible


def test_budget_cut_reprices_from_budget(client):
    thread_id, _, _ = _plan(client, NATURE_TRIP)
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "budget_cut", "target": "trip", "description": "Cut", "new_budget_inr": 40000})
    steps, after = _stream(client, thread_id)
    assert [s["agent"] for s in steps][:2] == ["critic_replanner", "budget_agent"]
    assert after["budget"]["ceiling_inr"] == 40000


def test_strike_replans_the_route(client):
    thread_id, _, before = _plan(client, NATURE_TRIP)
    client.post("/api/disrupt", json={"thread_id": thread_id, "type": "transport", "target": _cities(before)[0], "description": "Strike"})
    steps, _ = _stream(client, thread_id)
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
    thread_id, _, _ = _plan(client, NATURE_TRIP)
    assert client.post("/api/revise", json={"thread_id": thread_id, "message": message}).status_code == 200
    steps, done = _stream(client, thread_id)
    order = [s["agent"] for s in steps]
    assert order[0] == "edit_router" and order[1] == entry
    assert done["reran_from"] == entry


def test_a_lighter_day_has_one_sight(client):
    thread_id, _, _ = _plan(client, NATURE_TRIP)
    client.post("/api/revise", json={"thread_id": thread_id, "message": "make day 3 lighter"})
    _, done = _stream(client, thread_id)
    day3 = done["itinerary"]["days"][2]
    assert sum(1 for i in day3["items"] if i["kind"] == "activity") == 1


def test_a_question_leaves_the_plan_alone(client):
    thread_id, _, before = _plan(client, NATURE_TRIP)
    client.post("/api/revise", json={"thread_id": thread_id, "message": "which hotels?"})
    steps, done = _stream(client, thread_id)
    assert [s["agent"] for s in steps] == ["edit_router"]
    assert "Hotels:" in done["reply"]
    assert client.get(f"/api/itinerary/{thread_id}").json()["itinerary"] == before["itinerary"]


def test_health_says_there_is_no_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model"].startswith("none")
