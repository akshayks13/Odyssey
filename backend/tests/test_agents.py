"""Each agent's rules, one at a time."""
from __future__ import annotations

from datetime import date

import pytest

from agents.budget_agent import budget_agent_node, budget_tier
from agents.critic_replanner import critic_replanner_node
from agents.destination_agent import destination_agent_node
from agents.edit_router import parse_edit
from agents.mobility_agent import choose_mode, mobility_agent_node
from agents.trip_analyst import parse_request
from models.schemas import Disruption, DisruptionType, EditLocks, ReplanDirective
from orchestration.state import initial_state

TODAY = date(2026, 9, 27)


# --- Trip Analyst -------------------------------------------------------------

@pytest.mark.parametrize(
    "text, days, people, budget, pace, mode",
    [
        ("5 days in Kerala with 3 friends, around ₹40,000, nature and adventure at a relaxed pace", 5, 4, 40000, "relaxed", None),
        ("4-day Kerala trip for 2, by train, budget ₹35,000, beach and slow mornings", 4, 2, 35000, "relaxed", "rail"),
        ("A week in Munnar for a family of 5, 1.5 lakh, trekking, packed", 7, 5, 150000, "packed", None),
        ("weekend in Varkala solo 10k", 2, 1, 10000, "moderate", None),
        ("Kerala 3 nights ₹12,000 per person for 2 people", 4, 2, 24000, "moderate", None),
    ],
)
def test_parser_reads_the_request(text, days, people, budget, pace, mode):
    spec, parsed_mode = parse_request(text, TODAY)
    assert spec.destination_region == "Kerala, India"
    assert (spec.duration_days, spec.travellers, spec.budget_inr, spec.constraints.pace, parsed_mode) == (days, people, budget, pace, mode)


def test_parser_reads_interests_and_dates():
    spec, _ = parse_request("Kerala for 2 on 20 December, temples, museums and seafood", TODAY)
    prefs = spec.preferences.as_dict()
    assert prefs["culture"] == prefs["food"] == 0.9 and prefs["adventure"] < 0.5
    assert spec.start_date == "2026-12-20"
    assert parse_request("Kerala in January", TODAY)[0].start_date == "2027-01-10"  # next January


def test_parser_reports_what_it_assumed_and_asks_for_unknown_places():
    spec, _ = parse_request("Kerala please", TODAY)
    assert set(spec.needs_clarification) == {"duration", "travellers", "budget", "start_date"}
    spec, _ = parse_request("4 days in Delhi for 2", TODAY)
    assert spec.destination_region == "" and "Kerala" in spec.clarifying_question


# --- Destination ----------------------------------------------------------------

def _state(spec, **extra):
    s = initial_state(spec.raw_input)
    s.update(trip_spec=spec, **extra)
    return s


def test_destination_ranks_by_interest_and_keeps_stops_close(spec):
    out = destination_agent_node(_state(spec))
    names = [d.name for d in out["selected_destinations"]]
    assert len(names) == 3 and "Wayanad" not in names  # remote: over a day's drive from the others
    assert all(out["candidate_activities"][n] for n in names)


def test_destination_honours_named_cities_closures_and_the_critics_avoid_list():
    spec, _ = parse_request("6 days in Kochi and Varkala for 2 from 2027-01-10, ₹80,000, culture", TODAY)
    names = [d.name for d in destination_agent_node(_state(spec))["selected_destinations"]]
    assert names[:2] == ["Kochi", "Varkala"]

    closed = _state(spec, disruptions=[Disruption(type=DisruptionType.CLOSURE, target="Kochi", description="x")])
    assert "Kochi" not in [d.name for d in destination_agent_node(closed)["selected_destinations"]]

    sent_back = _state(spec, replan_directives=[ReplanDirective(target_agent="destination_agent", reason="far", constraints={"avoid": ["Varkala"]})])
    assert "Varkala" not in [d.name for d in destination_agent_node(sent_back)["selected_destinations"]]


def test_destination_replaces_only_the_closed_city(spec):
    first = destination_agent_node(_state(spec))["selected_destinations"]
    closed = first[0].name
    again = destination_agent_node(_state(
        spec,
        selected_destinations=first,
        disruptions=[Disruption(type=DisruptionType.CLOSURE, target=closed, description="x")],
        replan_directives=[ReplanDirective(target_agent="destination_agent", reason="closed", constraints={"issue": "CLOSURE"})],
    ))["selected_destinations"]
    assert closed not in [d.name for d in again]
    assert {d.name for d in first[1:]} <= {d.name for d in again}


# --- Mobility -------------------------------------------------------------------

def test_mode_rule():
    road, rail = {"hours": 3.0, "cost": 1000}, {"hours": 3.5, "cost": 300}
    assert choose_mode({"road": road, "rail": rail}, None, False, 4) == "road"
    assert choose_mode({"road": road, "rail": rail}, "rail", False, 4) == "rail"
    assert choose_mode({"road": road, "rail": rail}, None, True, 4) == "rail"  # cheapest, sent back over budget
    assert choose_mode({"road": {"hours": 5.0, "cost": 1000}, "rail": rail}, None, False, 4) == "rail"  # drive over the limit
    assert choose_mode({"road": road}, "rail", False, 4) == "road"  # no station: road


def _cities(spec, names):
    out = destination_agent_node(_state(spec, edit_locks=EditLocks(pinned_cities=names)))
    return [d for d in out["selected_destinations"] if d.name in names]


def test_mobility_orders_with_astar_and_takes_the_train_when_asked(spec):
    selected = _cities(spec, ["Kochi", "Alleppey", "Varkala"])
    out = mobility_agent_node(_state(spec, selected_destinations=selected, edit_locks=EditLocks(preferred_mode="rail")))
    route = out["route"]
    assert route.ordered_destinations in (["Kochi", "Alleppey", "Varkala"], ["Varkala", "Alleppey", "Kochi"])
    assert all(leg.mode.value == "rail" for leg in route.legs)
    assert route.search_algorithm.startswith("A*")


def test_a_strike_slows_the_road(spec):
    selected = _cities(spec, ["Munnar", "Thekkady"])
    calm = mobility_agent_node(_state(spec, selected_destinations=selected))["route"]
    strike = mobility_agent_node(_state(spec, selected_destinations=selected, disruptions=[Disruption(type=DisruptionType.TRANSPORT, target="Munnar", description="x")]))["route"]
    assert strike.total_duration_hours == pytest.approx(2 * calm.total_duration_hours)
    assert strike.total_cost_inr > calm.total_cost_inr


# --- Budget ---------------------------------------------------------------------

def _priced(spec, locks=None, **extra):
    s = _state(spec, edit_locks=locks or EditLocks(), **extra)
    s.update(destination_agent_node(s))
    s.update(mobility_agent_node(s))
    return s


def test_tier_from_budget_per_person():
    assert (budget_tier(5000), budget_tier(12000), budget_tier(30000)) == ("budget", "mid", "premium")


def test_budget_prefers_cheaper_hotels_before_dropping_sights(spec):
    tight = spec.model_copy(update={"budget_inr": 30000})
    out = budget_agent_node(_priced(tight))
    b = out["budget_breakdown"]
    assert all(h.tier == "budget" for h in b.selected_hotels)  # hotels were cut first


def test_budget_honours_a_hotel_the_traveller_chose(spec):
    out = budget_agent_node(_priced(spec, EditLocks(hotel_prefs={"*": "best"})))
    assert all(h.tier == "premium" for h in out["budget_breakdown"].selected_hotels)


def test_budget_goes_one_tier_cheaper_after_an_overrun(spec):
    s = _priced(spec)
    first = budget_agent_node(s)
    over = first["budget_breakdown"].model_copy(update={"total_inr": spec.budget_inr + 1})
    again = budget_agent_node({**s, "budget_breakdown": over})
    assert "one cheaper" in again["agent_meta"]["note"]
    assert again["budget_breakdown"].food_inr < first["budget_breakdown"].food_inr  # food is priced by tier too


# --- Critic ---------------------------------------------------------------------

def _critic_state(spec, **extra):
    from agents.itinerary_architect import itinerary_architect_node

    s = _priced(spec)
    s.update(budget_agent_node(s))
    s.update(itinerary_architect_node(s))
    s.update(extra)
    return s


@pytest.mark.parametrize(
    "disruption, target",
    [
        (DisruptionType.CLOSURE, "destination_agent"),
        (DisruptionType.WEATHER, "destination_agent"),
        (DisruptionType.TRANSPORT, "mobility_agent"),
    ],
)
def test_critic_routes_each_disruption_to_one_agent(spec, disruption, target):
    s = _critic_state(spec)
    city = s["selected_destinations"][0].name
    out = critic_replanner_node({**s, "disruptions": [Disruption(type=disruption, target=city, description="x")]})
    assert [d.target_agent for d in out["replan_directives"]] == [target]


def test_critic_routes_over_budget_to_budget_and_a_cut_to_budget(spec):
    s = _critic_state(spec)
    over = s["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    assert critic_replanner_node({**s, "budget_breakdown": over})["replan_directives"][0].target_agent == "budget_agent"
    cut = [Disruption(type=DisruptionType.BUDGET_CUT, target="trip", description="x", new_budget_inr=30000)]
    assert critic_replanner_node({**s, "disruptions": cut})["replan_directives"][0].constraints["issue"] == "BUDGET_CUT"


def test_critic_accepts_a_valid_plan_and_stops_at_the_iteration_limit(spec):
    s = _critic_state(spec)
    ok = critic_replanner_node(s)
    assert ok["validation_report"].valid and ok["final_itinerary"] is not None
    over = s["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    last = critic_replanner_node({**s, "budget_breakdown": over, "iteration_count": 2})
    assert last["replan_directives"] == [] and last["final_itinerary"] is not None


def test_critic_remembers_a_closed_city(spec):
    s = _critic_state(spec)
    city = s["selected_destinations"][0].name
    gone = [Disruption(type=DisruptionType.CLOSURE, target=city, description="x")]
    out = critic_replanner_node({**s, "disruptions": gone, "selected_destinations": s["selected_destinations"][1:], "draft_itinerary": None})
    assert city in out["edit_locks"].excluded_cities


# --- Edit Router ----------------------------------------------------------------

@pytest.mark.parametrize(
    "message, check",
    [
        ("make day 2 lighter", lambda p: p["lock_updates"]["light_days"] == [2]),
        ("day 3 free please", lambda p: p["lock_updates"]["free_days"] == [3]),
        ("add Varkala", lambda p: p["add_cities"] == ["Varkala"]),
        ("swap Munnar for Vagamon", lambda p: p["remove_cities"] == ["Munnar"] and p["add_cities"] == ["Vagamon"]),
        ("by train", lambda p: p["lock_updates"]["preferred_mode"] == "rail"),
        ("cheapest hotels", lambda p: p["lock_updates"]["hotel_prefs"] == {"*": "cheapest"}),
        ("make it 7 days", lambda p: p["spec_patch"]["duration_days"] == 7),
        ("budget ₹80,000", lambda p: p["spec_patch"]["budget_inr"] == 80000),
        ("more food", lambda p: p["spec_patch"]["preferences"] == {"food": 0.9}),
        ("start at 10", lambda p: p["lock_updates"]["day_start_hour"] == 10),
        ("Munnar is closed", lambda p: p["disruptions"][0]["type"] == "closure" and p["disruptions"][0]["target"] == "Munnar"),
        ("how much does it cost?", lambda p: p["intent"] == "answer" and "₹" in p["reply"]),
    ],
)
def test_edit_grammar(spec, message, check):
    s = _critic_state(spec)
    parsed = parse_edit(message, s)
    assert parsed and check(parsed), parsed


def test_edit_that_is_not_understood_changes_nothing(spec):
    assert parse_edit("blah blah", _critic_state(spec)) == {}
