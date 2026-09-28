"""Each agent's rules, one at a time. Agents are called directly with a message and a state."""
from __future__ import annotations

from datetime import date

import pytest

from agents.budget_agent import BudgetAgent, budget_tier
from agents.critic_replanner import CriticAgent
from agents.destination_agent import DestinationAgent
from agents.edit_router import parse_edit
from agents.itinerary_architect import ItineraryArchitect
from agents.mobility_agent import MobilityAgent, choose_mode
from agents.trip_analyst import parse_request
from config import MAX_REPLAN_ITERATIONS
from core.blackboard import initial_state
from core.messages import Message, MsgType
from core.strategies import STRATEGIES
from models.schemas import Disruption, DisruptionType, EditLocks, ReplanDirective

TODAY = date(2026, 9, 27)


def _state(spec, **extra):
    state = initial_state(spec.raw_input)
    state.update(trip_spec=spec, **extra)
    return state


def _run(agent_cls, state, msg_type=MsgType.RERUN, strategy="full", **payload):
    """Deliver one message to a fresh agent; returns its Result and applies its updates to `state`."""
    agent = agent_cls(STRATEGIES[strategy])
    result = agent.handle(Message(0, msg_type, "test", agent.name, "", payload), state)
    state.update(result.updates)
    return result


def _replan(agent: str, issue: str, avoid=(), **more):
    return ReplanDirective(target_agent=agent, reason=issue, constraints={"issue": issue, "avoid": list(avoid), **more})


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

def test_destination_ranks_by_interest_and_keeps_stops_close(spec):
    result = _run(DestinationAgent, _state(spec))
    names = [d.name for d in result.updates["selected_destinations"]]
    assert len(names) == 3 and "Wayanad" not in names  # remote: over a day's drive from the others
    assert all(result.updates["candidate_activities"][n] for n in names)
    assert [(p.type, p.recipient) for p in result.posts] == [(MsgType.CITIES_READY, "mobility_agent")]


def test_destination_honours_named_cities_closures_and_the_critics_avoid_list():
    spec, _ = parse_request("6 days in Kochi and Varkala for 2 from 2027-01-10, ₹80,000, culture", TODAY)
    names = [d.name for d in _run(DestinationAgent, _state(spec)).updates["selected_destinations"]]
    assert names[:2] == ["Kochi", "Varkala"]

    closed = _state(spec, disruptions=[Disruption(type=DisruptionType.CLOSURE, target="Kochi", description="x")])
    assert "Kochi" not in [d.name for d in _run(DestinationAgent, closed).updates["selected_destinations"]]

    told = _run(DestinationAgent, _state(spec), MsgType.REPLAN, directive=_replan("destination_agent", "TRAVEL_OVERLOAD", avoid=["Varkala"]))
    assert "Varkala" not in [d.name for d in told.updates["selected_destinations"]]


def test_destination_replaces_only_the_closed_city(spec):
    first = _run(DestinationAgent, _state(spec)).updates["selected_destinations"]
    closed = first[0].name
    state = _state(spec, selected_destinations=first, disruptions=[Disruption(type=DisruptionType.CLOSURE, target=closed, description="x")])
    again = _run(DestinationAgent, state, MsgType.REPLAN, directive=_replan("destination_agent", "CLOSURE")).updates["selected_destinations"]
    assert closed not in [d.name for d in again]
    assert {d.name for d in first[1:]} <= {d.name for d in again}


def test_a_restart_forgets_which_cities_were_fine(spec):
    first = _run(DestinationAgent, _state(spec)).updates["selected_destinations"]
    state = _state(spec, selected_destinations=first)
    told = _run(DestinationAgent, state, MsgType.REPLAN, directive=_replan("destination_agent", "LEISURE_DAY", restart=True))
    assert "kept" not in told.message  # it chose afresh, not "kept X"


# --- Mobility -------------------------------------------------------------------

def test_mode_rule():
    road, rail = {"hours": 3.0, "cost": 1000}, {"hours": 3.5, "cost": 300}
    assert choose_mode({"road": road, "rail": rail}, None, False, 4) == "road"
    assert choose_mode({"road": road, "rail": rail}, "rail", False, 4) == "rail"
    assert choose_mode({"road": road, "rail": rail}, None, True, 4) == "rail"  # cheapest, sent back over budget
    assert choose_mode({"road": {"hours": 5.0, "cost": 1000}, "rail": rail}, None, False, 4) == "rail"  # drive over the limit
    assert choose_mode({"road": road}, "rail", False, 4) == "road"  # no station: road


def _cities(spec, names):
    state = _state(spec, edit_locks=EditLocks(pinned_cities=names))
    return [d for d in _run(DestinationAgent, state).updates["selected_destinations"] if d.name in names]


def test_mobility_orders_with_astar_and_takes_the_train_when_asked(spec):
    state = _state(spec, selected_destinations=_cities(spec, ["Kochi", "Alleppey", "Varkala"]), edit_locks=EditLocks(preferred_mode="rail"))
    result = _run(MobilityAgent, state)
    route = result.updates["route"]
    assert route.ordered_destinations in (["Kochi", "Alleppey", "Varkala"], ["Varkala", "Alleppey", "Kochi"])
    assert all(leg.mode.value == "rail" for leg in route.legs)
    assert route.search_algorithm.startswith("A*")
    assert result.posts[0].type == MsgType.ROUTE_READY


def test_the_greedy_strategy_orders_by_nearest_neighbour(spec):
    state = _state(spec, selected_destinations=_cities(spec, ["Kochi", "Munnar", "Alleppey", "Varkala", "Thekkady"]))
    greedy = _run(MobilityAgent, dict(state), strategy="greedy_order").updates["route"]
    astar = _run(MobilityAgent, dict(state), strategy="full").updates["route"]
    assert greedy.search_algorithm == "Greedy nearest neighbour" and astar.search_algorithm.startswith("A*")
    assert greedy.total_duration_hours >= astar.total_duration_hours


def test_a_strike_slows_the_road(spec):
    selected = _cities(spec, ["Munnar", "Thekkady"])
    calm = _run(MobilityAgent, _state(spec, selected_destinations=selected)).updates["route"]
    strike = _run(MobilityAgent, _state(spec, selected_destinations=selected, disruptions=[Disruption(type=DisruptionType.TRANSPORT, target="Munnar", description="x")])).updates["route"]
    assert strike.total_duration_hours == pytest.approx(2 * calm.total_duration_hours)
    assert strike.total_cost_inr > calm.total_cost_inr


# --- Budget ---------------------------------------------------------------------

def _priced(spec, locks=None, **extra):
    state = _state(spec, edit_locks=locks or EditLocks(), **extra)
    _run(DestinationAgent, state)
    _run(MobilityAgent, state)
    return state


def test_tier_from_budget_per_person():
    assert (budget_tier(5000), budget_tier(12000), budget_tier(30000)) == ("budget", "mid", "premium")


def test_budget_prefers_cheaper_hotels_before_dropping_sights(spec):
    tight = spec.model_copy(update={"budget_inr": 30000})
    result = _run(BudgetAgent, _priced(tight))
    assert all(h.tier == "budget" for h in result.updates["budget_breakdown"].selected_hotels)  # hotels were cut first
    assert result.posts[0].type == MsgType.BUDGET_READY


def test_budget_honours_a_hotel_the_traveller_chose(spec):
    result = _run(BudgetAgent, _priced(spec, EditLocks(hotel_prefs={"*": "best"})))
    assert all(h.tier == "premium" for h in result.updates["budget_breakdown"].selected_hotels)


def test_budget_goes_one_tier_cheaper_after_an_overrun(spec):
    state = _priced(spec)
    first = _run(BudgetAgent, dict(state))
    over = first.updates["budget_breakdown"].model_copy(update={"total_inr": spec.budget_inr + 1})
    again = _run(BudgetAgent, {**state, "budget_breakdown": over})
    assert "one cheaper" in again.note
    assert again.updates["budget_breakdown"].food_inr < first.updates["budget_breakdown"].food_inr  # food is priced by tier too


# --- Architect ------------------------------------------------------------------

def _scheduled(spec, strategy="full", **extra):
    state = _priced(spec, **extra)
    _run(BudgetAgent, state)
    return state, _run(ItineraryArchitect, state, strategy=strategy)


def test_architect_hands_the_schedule_to_the_environment_only_when_observing(spec):
    _, observing = _scheduled(spec, "full")
    _, blind = _scheduled(spec, "static")
    assert observing.posts[0].recipient == "environment" and blind.posts[0].recipient == "critic_replanner"


def test_a_reported_heavy_rain_day_drops_outdoor_sights_but_keeps_them_for_later(spec):
    state, plain = _scheduled(spec)
    day = next(d for d in plain.updates["draft_itinerary"].days if any(i.category in ("nature", "adventure") for i in d.items))
    state["observed"] = state["observed"].model_copy(update={"heavy_rain": [f"{day.destination}|{day.date}"]})
    rainy = _run(ItineraryArchitect, state).updates["draft_itinerary"]
    same_day = next(d for d in rainy.days if d.day_number == day.day_number)
    assert not any(i.category in ("nature", "adventure") for i in same_day.items if i.kind == "activity")
    assert "Heavy rain reported" in (same_day.note or "") and same_day.weather.source == "field report"


def test_a_sight_reported_closed_is_not_scheduled_that_day(spec):
    state, plain = _scheduled(spec)
    day = next(d for d in plain.updates["draft_itinerary"].days if any(i.kind == "activity" for i in d.items))
    shut = next(i for i in day.items if i.kind == "activity")
    state["observed"] = state["observed"].model_copy(update={"closed": [f"{shut.activity_id}|{day.date}"]})
    again = _run(ItineraryArchitect, state).updates["draft_itinerary"]
    assert shut.activity_id not in [i.activity_id for i in again.days[day.day_number - 1].items]


# --- Critic ---------------------------------------------------------------------

def _critic_state(spec, **extra):
    state, _ = _scheduled(spec)
    state.update(extra)
    return state


@pytest.mark.parametrize(
    "disruption, target",
    [
        (DisruptionType.CLOSURE, "destination_agent"),
        (DisruptionType.WEATHER, "destination_agent"),
        (DisruptionType.TRANSPORT, "mobility_agent"),
    ],
)
def test_critic_routes_each_disruption_to_one_agent(spec, disruption, target):
    state = _critic_state(spec)
    city = state["selected_destinations"][0].name
    state["disruptions"] = [Disruption(type=disruption, target=city, description="x")]
    result = _run(CriticAgent, state, MsgType.DISRUPTION)
    assert [d.target_agent for d in result.updates["replan_directives"]] == [target]
    assert [(p.type, p.recipient) for p in result.posts] == [(MsgType.REPLAN, target)]
    assert result.posts[0].payload["directive"].target_agent == target


def test_critic_routes_over_budget_to_budget_and_a_cut_to_budget(spec):
    state = _critic_state(spec)
    over = state["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    assert _run(CriticAgent, {**state, "budget_breakdown": over}).updates["replan_directives"][0].target_agent == "budget_agent"
    cut = [Disruption(type=DisruptionType.BUDGET_CUT, target="trip", description="x", new_budget_inr=30000)]
    assert _run(CriticAgent, {**state, "disruptions": cut}).updates["replan_directives"][0].constraints["issue"] == "BUDGET_CUT"


def test_critic_accepts_a_valid_plan_and_stops_at_the_iteration_limit(spec):
    state = _critic_state(spec)
    ok = _run(CriticAgent, dict(state))
    assert ok.updates["validation_report"].valid and ok.updates["final_itinerary"] is not None
    assert [p.type for p in ok.posts] == [MsgType.ACCEPT]
    over = state["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    last = _run(CriticAgent, {**state, "budget_breakdown": over, "iteration_count": MAX_REPLAN_ITERATIONS - 1})
    assert last.updates["replan_directives"] == [] and last.updates["final_itinerary"] is not None
    assert [p.type for p in last.posts] == [MsgType.BEST_EFFORT]


def test_critic_remembers_a_closed_city(spec):
    state = _critic_state(spec)
    city = state["selected_destinations"][0].name
    gone = [Disruption(type=DisruptionType.CLOSURE, target=city, description="x")]
    result = _run(CriticAgent, {**state, "disruptions": gone, "selected_destinations": state["selected_destinations"][1:], "draft_itinerary": None})
    assert city in result.updates["edit_locks"].excluded_cities


def test_the_restart_strategy_always_sends_the_plan_to_the_destination_agent(spec):
    state = _critic_state(spec)
    over = state["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    result = _run(CriticAgent, {**state, "budget_breakdown": over}, strategy="restart")
    assert result.posts[0].recipient == "destination_agent" and result.posts[0].payload["directive"].constraints["restart"]


def test_the_static_strategy_reports_issues_but_never_replans(spec):
    state = _critic_state(spec)
    over = state["budget_breakdown"].model_copy(update={"total_inr": 10**6})
    result = _run(CriticAgent, {**state, "budget_breakdown": over}, strategy="static")
    assert result.updates["replan_directives"] == [] and [p.type for p in result.posts] == [MsgType.BEST_EFFORT]
    assert not result.updates["validation_report"].valid


# --- Edit Router ----------------------------------------------------------------

@pytest.mark.parametrize(
    "message, check",
    [
        ("make day 2 lighter", lambda p: p.lock_updates.light_days == [2]),
        ("day 3 free please", lambda p: p.lock_updates.free_days == [3]),
        ("add Varkala", lambda p: p.add_cities == ["Varkala"]),
        ("swap Munnar for Vagamon", lambda p: p.remove_cities == ["Munnar"] and p.add_cities == ["Vagamon"]),
        ("by train", lambda p: p.lock_updates.preferred_mode == "rail"),
        ("cheapest hotels", lambda p: p.lock_updates.hotel_prefs == {"*": "cheapest"}),
        ("make it 7 days", lambda p: p.spec_patch["duration_days"] == 7),
        ("budget ₹80,000", lambda p: p.spec_patch["budget_inr"] == 80000),
        ("more food", lambda p: p.spec_patch["preferences"] == {"food": 0.9}),
        ("start at 10", lambda p: p.lock_updates.day_start_hour == 10),
        ("Munnar is closed", lambda p: p.disruptions[0].type.value == "closure" and p.disruptions[0].target == "Munnar"),
        ("how much does it cost?", lambda p: p.intent == "answer" and "₹" in p.reply),
    ],
)
def test_edit_grammar(spec, message, check):
    parsed = parse_edit(message, _critic_state(spec))
    assert parsed and check(parsed), parsed


def test_edit_that_is_not_understood_changes_nothing(spec):
    assert parse_edit("blah blah", _critic_state(spec)) is None
