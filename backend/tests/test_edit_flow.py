"""Editing a finished plan by prompt. The Edit Router's model is scripted here, so these tests check what the
code does with a decision: which agent re-runs, that instructions persist, and that questions change nothing."""
from __future__ import annotations

import itertools
import json

import pytest
from fastapi.testclient import TestClient

from agents.critic_replanner import critic_replanner_node
from api.main import app
from models.schemas import BudgetBreakdown, Disruption, DisruptionType, Hotel, Itinerary, ItineraryDay, ReplanDirective, ScheduledItem, TripSpec
from orchestration.state import initial_state
from tests.conftest import assert_plan_is_coherent

_ids = itertools.count(1)
pytestmark = pytest.mark.usefixtures("use_scripted_analyst")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _stream(client, thread_id):
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as res:
        text = "".join(res.iter_text())
    events = [json.loads(line[5:]) for line in text.split("\n\n") if line.startswith("data:")]
    assert not [e for e in events if e["type"] == "error"], events
    return next(e for e in events if e["type"] == "done"), {e.get("agent") for e in events if e["type"] == "step_start"}


@pytest.fixture
def plan(client):
    thread_id = f"edit-{next(_ids)}"
    client.post("/api/plan", json={"message": "5 day trip to Kerala for 2 people, budget 60000, love nature and adventure, relaxed pace.", "thread_id": thread_id})
    done, _ = _stream(client, thread_id)
    assert_plan_is_coherent(done, why="the plan every edit test starts from")
    return thread_id, done


def _revise(client, thread_id, message):
    assert client.post("/api/revise", json={"thread_id": thread_id, "message": message}).status_code == 200
    return _stream(client, thread_id)


def _acts(day):
    return [i["activity_name"] for i in day["items"] if i["kind"] == "activity"]


def test_a_packed_day_reruns_only_the_scheduler(client, plan, scripted_llm):
    thread_id, _ = plan
    scripted_llm([{"intent": "modify", "route_to": "itinerary_architect", "summary": "Lighter day 2", "lock_updates": {"light_days": [2]}}])
    done, ran = _revise(client, thread_id, "day 2 is too packed")
    assert done["reran_from"] == "itinerary_architect" and done["summary"] == "Lighter day 2"
    assert {"itinerary_architect", "critic_replanner"} <= ran and not {"destination_agent", "mobility_agent", "budget_agent", "trip_analyst"} & ran
    day2 = next(d for d in done["itinerary"]["days"] if d["day_number"] == 2)
    assert day2["kind"] != "sightseeing" or len(_acts(day2)) <= 1


def test_adding_a_city_reenters_at_destination(client, plan, scripted_llm):
    thread_id, first = plan
    have = {d["name"] for d in first["selected_destinations"]}
    city = next(c for c in ("Alleppey", "Kochi", "Varkala", "Kovalam", "Wayanad") if c not in have)
    scripted_llm([{"intent": "modify", "route_to": "destination_agent", "summary": f"Add {city}", "add_cities": [city]}])
    done, _ = _revise(client, thread_id, f"add {city}")
    assert done["reran_from"] == "destination_agent" and city in done["route"]["ordered_destinations"]


def test_hotel_edit_reenters_at_budget(client, plan, scripted_llm):
    thread_id, first = plan
    scripted_llm([{"intent": "modify", "route_to": "budget_agent", "summary": "Nicer stays", "lock_updates": {"hotel_prefs": {"*": "best"}}}])
    done, ran = _revise(client, thread_id, "nicer hotels")
    assert done["reran_from"] == "budget_agent" and not {"destination_agent", "mobility_agent"} & ran
    before = {h["destination"]: h["rating"] for h in first["budget"]["selected_hotels"]}
    assert all(h["rating"] >= before[h["destination"]] for h in done["budget"]["selected_hotels"] if h["destination"] in before)


def test_the_router_cannot_skip_an_agent_whose_inputs_changed(client, plan, scripted_llm):
    thread_id, first = plan
    have = {d["name"] for d in first["selected_destinations"]}
    city = next(c for c in ("Alleppey", "Kochi", "Varkala", "Kovalam") if c not in have)
    scripted_llm([{"intent": "modify", "route_to": "itinerary_architect", "summary": "Add a stop", "add_cities": [city]}])
    assert _revise(client, thread_id, f"add {city}")[0]["reran_from"] == "destination_agent"


def test_the_router_may_rerun_more_than_needed(client, plan, scripted_llm):
    thread_id, _ = plan
    scripted_llm([{"intent": "modify", "route_to": "destination_agent", "summary": "Lighter", "lock_updates": {"light_days": [2]}}])
    assert _revise(client, thread_id, "easier day 2")[0]["reran_from"] == "destination_agent"


def test_a_question_changes_nothing(client, plan, scripted_llm):
    thread_id, first = plan
    scripted_llm([{"intent": "answer", "route_to": "answer", "reply": "They best match nature and adventure."}])
    done, ran = _revise(client, thread_id, "why these cities?")
    assert done["reply"] == "They best match nature and adventure." and ran == {"edit_router"}
    assert done["itinerary"]["days"] == first["itinerary"]["days"]


def test_names_that_are_not_in_the_plan_are_reported(client, plan, scripted_llm):
    thread_id, first = plan
    scripted_llm([{"intent": "modify", "route_to": "itinerary_architect", "summary": "Skip it", "lock_updates": {"excluded_activities": ["Nonexistent Waterpark"]}}])
    done, _ = _revise(client, thread_id, "skip the waterpark")
    assert "Nonexistent Waterpark" in done["reply"] and done["itinerary"]["days"] == first["itinerary"]["days"]


def test_without_a_model_the_plan_is_left_alone(client, plan, monkeypatch):
    thread_id, first = plan
    monkeypatch.setattr("agents.edit_router.get_llm", lambda: None)
    done, _ = _revise(client, thread_id, "make it 7 days")
    assert "language model" in done["reply"] and done["trip"]["duration_days"] == first["trip"]["duration_days"]


def test_a_disruption_in_prose_goes_to_the_critic(client, plan, scripted_llm):
    thread_id, first = plan
    city = first["selected_destinations"][0]["name"]
    scripted_llm([{"intent": "modify", "route_to": "critic_replanner", "summary": f"{city} closed",
                   "disruptions": [{"type": "closure", "target": city, "description": "landslide"}]}])
    done, ran = _revise(client, thread_id, f"the road to {city} is closed")
    assert done["reran_from"] == "critic_replanner" and "destination_agent" in ran
    assert city not in {d["destination"] for d in done["itinerary"]["days"]}


def test_earlier_instructions_survive_later_edits(client, plan, scripted_llm):
    thread_id, first = plan
    victim = next(a for d in first["itinerary"]["days"] for a in _acts(d))
    scripted_llm([
        {"intent": "modify", "route_to": "itinerary_architect", "summary": "Skip it", "lock_updates": {"excluded_activities": [victim]}},
        {"intent": "modify", "route_to": "budget_agent", "summary": "Cheaper", "lock_updates": {"hotel_prefs": {"*": "cheapest"}}},
    ])
    _revise(client, thread_id, f"skip {victim}")
    done, _ = _revise(client, thread_id, "cheaper hotels")
    assert victim not in [a for d in done["itinerary"]["days"] for a in _acts(d)], "a skipped sight must stay skipped"


def test_editing_works_after_a_restart(client, plan, scripted_llm):
    thread_id, first = plan
    import orchestration.graph as graph_module

    graph_module._compiled_graph = None  # a new process has an empty in-memory checkpointer
    scripted_llm([{"intent": "modify", "route_to": "itinerary_architect", "summary": "Lighter", "lock_updates": {"light_days": [2]}}])
    done, _ = _revise(client, thread_id, "day 2 lighter")
    assert done["reran_from"] == "itinerary_architect" and len(done["itinerary"]["days"]) == len(first["itinerary"]["days"])


def test_reconnecting_the_stream_after_a_restart_recovers_the_plan(client, plan):
    """GET /api/plan/{id}/stream used to only look at the in-memory checkpoint (or a freshly
    queued /api/plan). After a restart, with nothing pending and an empty checkpoint, a thread
    with a perfectly good saved plan 404'd instead of resuming it — /api/disrupt already had
    this exact fallback; the stream route needs it too."""
    import orchestration.graph as graph_module

    thread_id, first = plan
    graph_module._compiled_graph = None  # a new process has an empty in-memory checkpointer
    res = client.get(f"/api/plan/{thread_id}/stream")
    assert res.status_code == 200
    events = [json.loads(line[5:]) for line in res.text.split("\n\n") if line.startswith("data:")]
    assert not [e for e in events if e["type"] == "error"], events
    done = next(e for e in events if e["type"] == "done")
    assert done["itinerary"]["days"] == first["itinerary"]["days"]


def test_get_itinerary_prefers_the_saved_plan_over_a_partial_checkpoint(client, plan):
    """A reload used to read the live in-memory checkpoint first. After a crash mid-edit, that
    checkpoint can be genuinely partial (no itinerary yet) even though a good plan is saved --
    the reload must not show a blank page when a perfectly good plan is on disk."""
    from orchestration.graph import get_graph

    thread_id, first = plan
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"final_itinerary": None, "draft_itinerary": None}, as_node="itinerary_architect")
    res = client.get(f"/api/itinerary/{thread_id}")
    assert res.status_code == 200
    assert res.json()["itinerary"]["days"] == first["itinerary"]["days"]


def test_the_disruption_buttons_still_replan(client, plan):
    thread_id, first = plan
    city = first["selected_destinations"][0]["name"]
    assert client.post("/api/disrupt", json={"thread_id": thread_id, "type": "closure", "target": city, "description": "closed"}).status_code == 200
    done, _ = _stream(client, thread_id)
    assert city not in {d["destination"] for d in done["itinerary"]["days"]}
    assert_plan_is_coherent(done, why="after a closure was injected")


def test_revising_an_unknown_plan_is_a_404(client):
    assert client.post("/api/revise", json={"thread_id": "nope-nope", "message": "hi"}).status_code == 404


# --- the Critic feeds back to the agent that can fix a problem ---------------------------------------------------

def _critic_state(days, **extra):
    spec = TripSpec(destination_region="X", duration_days=len(days), budget_inr=100000, travellers=2)
    state = initial_state("x")
    state.update(trip_spec=spec, draft_itinerary=Itinerary(days=days, optimization_score=0.5), budget_breakdown=BudgetBreakdown(total_inr=1000, ceiling_inr=100000), **extra)
    return state


def _day(n, kind="sightseeing", acts=1, hotel=True):
    items = [ScheduledItem(activity_id=f"a{n}{i}", activity_name=f"S{n}{i}", destination="X", start_hour=9 + 3 * i, end_hour=11 + 3 * i) for i in range(acts)]
    stay = Hotel(name="Inn", destination="X", price_per_night_inr=2000) if hotel else None
    return ItineraryDay(day_number=n, destination="X", kind=kind, items=items, overnight_hotel=stay)


def test_a_day_with_nothing_left_to_see_goes_back_to_destination_with_the_reason():
    out = critic_replanner_node(_critic_state([_day(1), _day(2, kind="leisure", acts=0)]))
    directive = out["replan_directives"][0]
    assert directive.target_agent == "destination_agent" and "Day 2" in directive.reason
    assert directive.constraints["issue"] == "LEISURE_DAY"


def test_the_destination_agent_is_told_why_it_was_sent_back(planning_state, monkeypatch):
    import agents.destination_agent as destination

    seen = []
    monkeypatch.setattr(destination, "get_llm", lambda: object())
    monkeypatch.setattr(destination, "llm_decide", lambda llm, tools, system, user, **k: seen.append(user) or {})
    state = {**planning_state, "replan_directives": [ReplanDirective(target_agent="destination_agent", reason="Day 3 has nothing left to see")]}
    destination.destination_agent_node(state)
    assert any("Day 3 has nothing left to see" in u for u in seen)


def test_mobility_is_told_why_it_was_sent_back(planning_state, monkeypatch):
    import agents.mobility_agent as mobility

    seen = []
    monkeypatch.setattr(mobility, "get_llm", lambda: object())
    monkeypatch.setattr(mobility, "llm_decide", lambda llm, tools, system, user, **k: seen.append(user) or {})
    state = {**planning_state, "selected_destinations": [__import__("models.schemas", fromlist=["Destination"]).Destination(
        name="Munnar", coordinates={"lat": 10.09, "lng": 77.06})],
        "replan_directives": [ReplanDirective(target_agent="mobility_agent", reason="Pushkar → Mumbai is 19h by road")]}
    mobility.mobility_agent_node(state)
    assert any("19h by road" in u for u in seen)


def test_an_unavailable_quote_is_asked_again_not_remembered(monkeypatch):
    import tools.travel_market as market

    market._transport_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"available": False, "note": "no service"})
    assert not market.search_public_transport.invoke({"origin": "Pushkar", "destination": "Mumbai", "mode": "air"})["available"]
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"available": True, "operator": "IndiGo", "duration_hours": 5, "price_inr": 6000})
    assert market.search_public_transport.invoke({"origin": "Pushkar", "destination": "Mumbai", "mode": "air"})["available"]
    market._transport_cache.clear()


def test_an_issue_that_was_already_retried_is_not_sent_again():
    state = _critic_state([_day(1), _day(2, kind="leisure", acts=0)], iteration_count=1,
                          replan_directives=[ReplanDirective(target_agent="destination_agent", reason="r", constraints={"issue": "LEISURE_DAY"})])
    out = critic_replanner_node(state)
    assert not out["replan_directives"] and out["final_itinerary"] is not None


def test_a_free_day_the_user_asked_for_is_not_an_issue():
    from models.schemas import EditLocks

    out = critic_replanner_node(_critic_state([_day(1), _day(2, kind="leisure", acts=0)], edit_locks=EditLocks(free_days=[2])))
    assert out["validation_report"].valid


def test_a_night_without_a_hotel_goes_back_to_budget():
    days = [_day(1, hotel=False), _day(2)]
    out = critic_replanner_node(_critic_state(days))
    assert out["replan_directives"][0].target_agent == "budget_agent" and "No hotel" in out["replan_directives"][0].reason


def test_an_edit_does_not_send_the_plan_back_to_pick_new_cities():
    from models.schemas import EditDirective

    state = _critic_state([_day(1), _day(2, kind="leisure", acts=0)], edit_directive=EditDirective(intent="modify", entry="itinerary_architect"))
    out = critic_replanner_node(state)
    assert not out["replan_directives"], "the traveller asked for one change, not a new set of cities"


def test_a_long_road_trip_to_get_there_goes_back_to_mobility():
    from models.schemas import Route, RouteLeg

    state = _critic_state([_day(1), _day(2)])
    state["trip_spec"].origin_city = "Chennai"
    state["route"] = Route(ordered_destinations=["X"], legs=[RouteLeg(origin="Chennai", destination="X", mode="road", duration_hours=40.1, cost_inr=16000)],
                           return_leg=RouteLeg(origin="X", destination="Chennai", mode="road", duration_hours=40.1, cost_inr=16000))
    out = critic_replanner_node(state)
    assert out["replan_directives"][0].target_agent == "mobility_agent" and "40h by road" in out["replan_directives"][0].reason
    state["edit_locks"] = __import__("models.schemas", fromlist=["EditLocks"]).EditLocks(preferred_mode="road")
    assert not any(i.type == "LONG_ROAD_TRIP" for i in critic_replanner_node(state)["validation_report"].issues)  # they chose road


def test_a_hop_with_no_route_goes_back_to_destination_instead_of_passing_as_zero_hours():
    """Seen live: two Antarctic stations 3,800 km apart came back as a 0.0h, Rs 0 road hop and the plan was called valid."""
    from models.schemas import Route, RouteLeg

    state = _critic_state([_day(1), _day(2)])
    state["route"] = Route(ordered_destinations=["A", "B"], legs=[RouteLeg(origin="A", destination="B", mode="road", duration_hours=0.0, cost_inr=0, source="unavailable")])
    out = critic_replanner_node(state)
    directive = out["replan_directives"][0]
    assert not out["validation_report"].valid and directive.target_agent == "destination_agent"
    assert directive.constraints["issue"] == "NO_ROUTE" and directive.constraints["avoid"] == ["B"]


def test_a_reported_closure_is_handled_before_an_unrelated_budget_problem():
    """Seen live: after a landslide closed Munnar, an over-budget flag came first, iterations ran out, and Munnar stayed in the plan."""
    state = _critic_state([_day(1), _day(2)])
    state["budget_breakdown"] = BudgetBreakdown(total_inr=150000, ceiling_inr=100000)
    state["disruptions"] = [Disruption(type=DisruptionType.CLOSURE, target="X", description="landslide", day=1)]
    directive = critic_replanner_node(state)["replan_directives"][0]
    assert directive.constraints["issue"] == "CLOSURE" and directive.target_agent == "destination_agent"


def test_an_edit_never_re_picks_the_cities_because_a_hotel_lookup_failed():
    from models.schemas import EditDirective, ReplanDirective

    state = _critic_state([_day(1, hotel=False), _day(2)], edit_directive=EditDirective(intent="modify", entry="mobility_agent"))
    state["replan_directives"] = [ReplanDirective(target_agent="budget_agent", reason="No hotel found for X.", constraints={"issue": "NO_HOTEL", "avoid": []})]
    assert not critic_replanner_node(state)["replan_directives"]


def test_a_trip_that_is_all_travel_is_flagged_and_goes_to_mobility():
    """Seen live: 2 days in Kochi from Chennai by a 13.5h train each way left no sight at all, and the plan was called valid."""
    state = _critic_state([_day(1, kind="travel", acts=0), _day(2, kind="travel", acts=0)])
    out = critic_replanner_node(state)
    assert not out["validation_report"].valid
    assert out["replan_directives"][0].target_agent == "mobility_agent" and out["replan_directives"][0].constraints["issue"] == "ALL_TRAVEL"


def test_a_budget_eaten_by_fares_goes_to_mobility_and_a_hotel_heavy_one_to_budget():
    """Seen live: 15,000 rupees for Chennai-Kochi, with 16,000 of flights; Budget cannot cut its way out of that."""
    from models.schemas import Route, RouteLeg

    state = _critic_state([_day(1), _day(2)])
    state["budget_breakdown"] = BudgetBreakdown(total_inr=18000, ceiling_inr=15000)
    state["route"] = Route(ordered_destinations=["X"], total_cost_inr=16000, legs=[RouteLeg(origin="Chennai", destination="X", mode="air", duration_hours=4, cost_inr=16000)])
    assert critic_replanner_node(state)["replan_directives"][0].target_agent == "mobility_agent"
    state["route"] = Route(ordered_destinations=["X"], total_cost_inr=1000, legs=[])
    assert critic_replanner_node(state)["replan_directives"][0].target_agent == "budget_agent"


def test_no_hotel_after_a_retry_asks_destination_for_a_nearby_base_and_not_the_same_city_again(planning_state, monkeypatch):
    from models.schemas import ReplanDirective

    state = _critic_state([_day(1, hotel=False), _day(2)])
    state["replan_directives"] = [ReplanDirective(target_agent="budget_agent", reason="No hotel found for X.", constraints={"issue": "NO_HOTEL", "avoid": []})]
    directive = critic_replanner_node(state)["replan_directives"][0]
    assert directive.target_agent == "destination_agent" and directive.constraints["issue"] == "NO_BASE" and directive.constraints["avoid"] == ["X"]

    import agents.destination_agent as destination
    monkeypatch.setattr(destination, "llm_decide", lambda *a, **k: {})
    spec = planning_state["trip_spec"]
    first = destination.destination_agent_node(planning_state)["selected_destinations"][0].name
    planning_state["replan_directives"] = [ReplanDirective(target_agent="destination_agent", reason="no hotels", constraints={"issue": "NO_BASE", "avoid": [first]})]
    again = [d.name for d in destination.destination_agent_node(planning_state)["selected_destinations"]]
    assert first not in again, "the city without hotels must not be chosen again"


def test_a_sight_listed_under_two_stops_is_visited_once():
    """Seen live: 'Fort Kochi' and 'Kochi' both listed the Chinese Fishing Nets and Mattancherry Palace, so day 2 repeated day 1."""
    import agents.destination_agent as destination

    same = [{"id": "s1", "name": "Chinese Fishing Nets", "category": "culture", "duration_minutes": 60, "cost_inr": 0, "rating": 4.5, "opening_hour": 8, "closing_hour": 18,
             "coordinates": {"lat": 9.96, "lng": 76.24}}]
    state = initial_state("x")
    spec = TripSpec(destination_region="Kerala, India", duration_days=4, budget_inr=90000, travellers=2, raw_input="4 days in Kerala")
    state["trip_spec"] = spec
    import tools.foursquare_api as fsq
    fsq._attraction_cache.clear()
    orig = destination.search_attractions
    destination.search_attractions = type("T", (), {"invoke": staticmethod(lambda a: list(same))})
    try:
        out = destination.destination_agent_node(state)
    finally:
        destination.search_attractions = orig
    assert len(out["candidate_activities"]) >= 2, "the point is two stops offering the same sight"
    names = [a.name for acts in out["candidate_activities"].values() for a in acts]
    assert names.count("Chinese Fishing Nets") == 1


# =========================================================================================================================
# Every kind of edit, with its effect on the plan. The router's model is scripted; what is under test is that each
# instruction (a) re-enters at the right agent and (b) actually changes the plan the way it says.
# =========================================================================================================================

def _all_acts(done):
    return [a for d in done["itinerary"]["days"] for a in _acts(d)]


def _days(done):
    return done["itinerary"]["days"]


def _modify(route_to, summary="Edit", **fields):
    return {"intent": "modify", "route_to": route_to, "summary": summary, **fields}


EDIT_CASES = {}


def edit_case(name):
    def register(fn):
        EDIT_CASES[name] = fn
        return fn
    return register


@edit_case("pace relaxed: at most two sights a day")
def _(first):
    return _modify("itinerary_architect", spec_patch={"pace": "relaxed"}), lambda done: all(len(_acts(d)) <= 2 for d in _days(done))


@edit_case("pace packed: fits at least as much as before")
def _(first):
    return _modify("itinerary_architect", spec_patch={"pace": "packed"}), lambda done: len(_all_acts(done)) >= len(_all_acts(first))


@edit_case("light day: that day has at most one sight")
def _(first):
    return _modify("itinerary_architect", lock_updates={"light_days": [3]}), lambda done: len(_acts(_days(done)[2])) <= 1


@edit_case("free day: that day is free time with nothing scheduled")
def _(first):
    return _modify("itinerary_architect", lock_updates={"free_days": [2]}), lambda done: _days(done)[1]["kind"] == "leisure" and not _acts(_days(done)[1])


@edit_case("later start: nothing before 11:00")
def _(first):
    return _modify("itinerary_architect", lock_updates={"day_start_hour": 11}), lambda done: all(
        i["start_hour"] >= 11 for d in _days(done) for i in d["items"] if i["kind"] == "activity")


@edit_case("skip a sight")
def _(first):
    victim = _all_acts(first)[0]
    return _modify("itinerary_architect", lock_updates={"excluded_activities": [victim]}), lambda done: victim not in _all_acts(done)


@edit_case("move a sight to another day")
def _(first):
    days = _days(first)
    day_a = next(d for d in days if _acts(d))
    target = next((d for d in days if d["day_number"] != day_a["day_number"] and d["destination"] == day_a["destination"] and d["kind"] == "sightseeing"), None)
    if target is None:
        pytest.skip("this plan has only one sightseeing day in that city")
    name = _acts(day_a)[0]
    return (_modify("itinerary_architect", lock_updates={"pinned_activities": {name: target["day_number"]}}),
            lambda done: name in _acts(_days(done)[target["day_number"] - 1]))


@edit_case("cheapest hotels: no dearer than before")
def _(first):
    prices = {h["destination"]: h["price_per_night_inr"] for h in first["budget"]["selected_hotels"]}
    return _modify("budget_agent", lock_updates={"hotel_prefs": {"*": "cheapest"}}), lambda done: all(
        h["price_per_night_inr"] <= prices.get(h["destination"], 1e9) for h in done["budget"]["selected_hotels"])


@edit_case("best hotels: rated no lower than before")
def _(first):
    ratings = {h["destination"]: h["rating"] for h in first["budget"]["selected_hotels"]}
    return _modify("budget_agent", lock_updates={"hotel_prefs": {"*": "best"}}), lambda done: all(
        h["rating"] >= ratings.get(h["destination"], 0) for h in done["budget"]["selected_hotels"])


@edit_case("a named hotel in one city")
def _(first):
    city = first["selected_destinations"][0]["name"]
    options = [h for h in first["accommodation_options"] if h["destination"] == city]
    wanted = max(options, key=lambda h: h["price_per_night_inr"])
    return (_modify("budget_agent", lock_updates={"hotel_prefs": {city.lower(): wanted["name"]}}),
            lambda done: next(h for h in done["budget"]["selected_hotels"] if h["destination"] == city)["name"] == wanted["name"])


@edit_case("travel by rail")
def _(first):
    return _modify("mobility_agent", lock_updates={"preferred_mode": "rail"}), lambda done: done["route"]["legs"] and all(l["mode"] == "rail" for l in done["route"]["legs"])


@edit_case("travel by road")
def _(first):
    return _modify("mobility_agent", lock_updates={"preferred_mode": "road"}), lambda done: all(l["mode"] == "road" for l in done["route"]["legs"])


@edit_case("add a city")
def _(first):
    have = {d["name"] for d in first["selected_destinations"]}
    city = next(c for c in ("Alleppey", "Kochi", "Varkala", "Kovalam", "Wayanad") if c not in have)
    return _modify("destination_agent", add_cities=[city]), lambda done: city in done["route"]["ordered_destinations"] and len(_days(done)) == len(_days(first))


@edit_case("remove a city")
def _(first):
    city = first["route"]["ordered_destinations"][-1]
    return _modify("destination_agent", remove_cities=[city]), lambda done: city not in done["route"]["ordered_destinations"] and len(_days(done)) == len(_days(first))


@edit_case("longer trip: 7 days")
def _(first):
    return _modify("budget_agent", spec_patch={"duration_days": 7}), lambda done: len(_days(done)) == 7 and done["trip"]["duration_days"] == 7


@edit_case("shorter trip: 3 days")
def _(first):
    return _modify("budget_agent", spec_patch={"duration_days": 3}), lambda done: len(_days(done)) == 3


@edit_case("lower budget is the new ceiling")
def _(first):
    return _modify("budget_agent", spec_patch={"budget_inr": 30000}), lambda done: done["budget"]["ceiling_inr"] == 30000


@edit_case("higher budget is the new ceiling")
def _(first):
    return _modify("budget_agent", spec_patch={"budget_inr": 200000}), lambda done: done["budget"]["ceiling_inr"] == 200000 and done["budget"]["over_budget_by_inr"] == 0 and not any(i["type"] == "BUDGET" for i in done["issues"])


@edit_case("more travellers: dearer plan")
def _(first):
    return _modify("mobility_agent", spec_patch={"travellers": 6}), lambda done: done["trip"]["travellers"] == 6 and done["budget"]["total_inr"] > first["budget"]["total_inr"]


@edit_case("new start date moves every day")
def _(first):
    return _modify("mobility_agent", spec_patch={"start_date": "2099-03-01"}), lambda done: [d["date"] for d in _days(done)][0] == "2099-03-01" and _days(done)[-1]["date"] == "2099-03-05"


@edit_case("arriving from somewhere adds the way home")
def _(first):
    return _modify("destination_agent", spec_patch={"origin_city": "Delhi"}), lambda done: done["route"]["return_leg"] is not None


@edit_case("changed tastes keep the same cities")
def _(first):
    return (_modify("destination_agent", spec_patch={"preferences": {"food": 0.9, "adventure": 0.1}}),
            lambda done: done["route"]["ordered_destinations"] == first["route"]["ordered_destinations"])


@edit_case("closure of a city: it disappears from the plan")
def _(first):
    city = first["selected_destinations"][0]["name"]
    return (_modify("critic_replanner", disruptions=[{"type": "closure", "target": city, "description": "closed"}]),
            lambda done: city not in {d["destination"] for d in _days(done)})


@edit_case("weather alert on a city: it is replaced")
def _(first):
    city = first["selected_destinations"][0]["name"]
    return (_modify("critic_replanner", disruptions=[{"type": "weather", "target": city, "description": "cyclone"}]),
            lambda done: city not in {d["destination"] for d in _days(done)})


@edit_case("transport strike: journeys to that city get slower")
def _(first):
    city = first["route"]["ordered_destinations"][-1]
    before = sum(l["duration_hours"] for l in first["route"]["legs"] if city in (l["origin"], l["destination"]))
    return (_modify("critic_replanner", disruptions=[{"type": "transport", "target": city, "description": "strike"}]),
            lambda done: sum(l["duration_hours"] for l in done["route"]["legs"] if city in (l["origin"], l["destination"])) > before)


@edit_case("budget cut becomes the ceiling")
def _(first):
    return (_modify("critic_replanner", disruptions=[{"type": "budget_cut", "target": "trip", "description": "cut", "new_budget_inr": 20000}]),
            lambda done: done["budget"]["ceiling_inr"] == 20000)


@pytest.mark.parametrize("name", list(EDIT_CASES))
def test_every_kind_of_edit_changes_the_plan_as_asked(client, plan, scripted_llm, monkeypatch, name):
    import tools.travel_market as market

    market._transport_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"available": True, "operator": "Indian Railways", "duration_hours": 4.0, "price_inr": 500})
    thread_id, first = plan
    decision, check = EDIT_CASES[name](first)
    scripted_llm([decision])
    done, _ = _revise(client, thread_id, name)
    assert done.get("reply") is None, done.get("reply")
    shown = [(d["day_number"], d["kind"], [(i["activity_name"], i["start_hour"]) for i in d["items"] if i["kind"] == "activity"]) for d in done["itinerary"]["days"]]
    assert check(done), f"'{name}' did not change the plan as asked: {shown}"
    # Doing what was asked is not enough: the rest of the plan has to survive the change.
    assert_plan_is_coherent(done, why=f"after edit: {name}")
    market._transport_cache.clear()


def test_the_sample_trip_opens_and_can_be_edited_with_no_model(monkeypatch):
    """The demo path: no model at all, yet the sample opens, and an edit says plainly that it cannot run."""
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setattr("llm._providers", lambda: [])
    client = TestClient(app)
    thread_id = client.post("/api/sample").json()["thread_id"]
    plan = client.get(f"/api/itinerary/{thread_id}").json()
    assert len(plan["itinerary"]["days"]) == 4 and plan["issues"] == []
    assert all(d["items"] for d in plan["itinerary"]["days"])

    assert client.post("/api/revise", json={"thread_id": thread_id, "message": "make day 2 lighter"}).status_code == 200
    with client.stream("GET", f"/api/plan/{thread_id}/stream") as res:
        events = [line for line in res.iter_lines() if line.startswith("data:")]
    assert "language model" in events[-1] and "unchanged" in events[-1]
    assert client.get(f"/api/itinerary/{thread_id}").json()["itinerary"]["days"], "the plan is still there after a failed edit"


def test_a_disruption_works_on_a_saved_plan_after_a_restart(client, plan):
    """The disruption buttons must not depend on the plan still being in server memory."""
    from orchestration.graph import get_graph

    thread_id, _ = plan
    fresh = client.post("/api/sample").json()["thread_id"]  # saved in SQLite only, never planned in this process
    res = client.post("/api/disrupt", json={"thread_id": fresh, "type": "closure", "target": "Munnar", "description": "closed", "day": 1})
    assert res.status_code == 200
    saved = get_graph().get_state({"configurable": {"thread_id": fresh}}).values
    assert [d.target for d in saved["disruptions"]] == ["Munnar"] and saved["selected_destinations"]
    assert client.post("/api/disrupt", json={"thread_id": "no-such-plan", "type": "closure", "target": "X", "description": "x"}).status_code == 404
