"""The multi-agent runtime: messages, ownership of state, the field, and the strategies."""
from __future__ import annotations

from datetime import date

import pytest

from core.blackboard import AccessError, Blackboard
from core.compare import compare_strategies, run_strategy
from core.environment import FieldWorld
from core.messages import MsgType
from core.strategies import STRATEGIES
from core.team import Team
from models.schemas import Disruption, DisruptionType
from tests.conftest import NATURE_TRIP, RAINY_TRIP


def _chain(team: Team) -> list[tuple[str, str, MsgType]]:
    return [(m.sender, m.recipient, m.type) for m in team.bus.log]


# --- messages ---------------------------------------------------------------------

def test_a_plan_is_a_chain_of_addressed_messages(make_team):
    assert _chain(make_team()) == [
        ("traveller", "trip_analyst", MsgType.REQUEST),
        ("trip_analyst", "destination_agent", MsgType.SPEC_READY),
        ("destination_agent", "mobility_agent", MsgType.CITIES_READY),
        ("mobility_agent", "budget_agent", MsgType.ROUTE_READY),
        ("budget_agent", "itinerary_architect", MsgType.BUDGET_READY),
        ("itinerary_architect", "critic_replanner", MsgType.SCHEDULE_READY),
        ("critic_replanner", "traveller", MsgType.ACCEPT),
    ]


def test_with_the_field_on_the_schedule_goes_through_the_environment_first(make_team):
    team = make_team(uncertainty="normal")
    chain = _chain(team)
    assert ("itinerary_architect", "environment", MsgType.SCHEDULE_READY) in chain
    assert ("environment", "critic_replanner", MsgType.FIELD_REPORT) in chain


def test_a_question_to_the_traveller_ends_the_run(make_team):
    team = make_team("4 days in Delhi for 2, ₹50,000")
    assert _chain(team)[-1] == ("trip_analyst", "traveller", MsgType.NEED_INFO)
    assert team.bb["draft_itinerary"] is None


def test_a_reported_closure_goes_to_the_critic_and_then_to_one_owner(make_team):
    team = make_team()
    city = team.bb["selected_destinations"][0].name
    before = len(team.bus.log)
    team.run_to_end(team.disrupt(Disruption(type=DisruptionType.CLOSURE, target=city, description="road closed")))
    new = [(m.sender, m.recipient, m.type) for m in team.bus.log[before:]]
    assert new[0] == ("traveller", "critic_replanner", MsgType.DISRUPTION)
    assert new[1] == ("critic_replanner", "destination_agent", MsgType.REPLAN)
    assert [m for m in new if m[2] == MsgType.REPLAN and m[1] != "destination_agent"] == []  # nobody else was told
    assert city not in [d.name for d in team.bb["selected_destinations"]]


def test_an_edit_goes_to_the_router_which_re_enters_at_the_earliest_affected_agent(make_team):
    team = make_team()
    before = len(team.bus.log)
    team.run_to_end(team.edit("make day 2 lighter"))
    new = [(m.sender, m.recipient, m.type) for m in team.bus.log[before:]]
    assert new[:2] == [("traveller", "edit_router", MsgType.EDIT), ("edit_router", "itinerary_architect", MsgType.RERUN)]
    team.run_to_end(team.edit("how much does it cost?"))
    assert team.bus.log[-1].type == MsgType.ANSWER and team.bus.log[-1].recipient == "traveller"


def test_the_ui_sees_every_message_and_every_step(make_team):
    team = Team("full", 1, "off")
    events = list(team.plan(NATURE_TRIP))
    assert [e["kind"] for e in events if e["type"] == "message"] == [m.type.value for m in team.bus.log]
    assert [e["agent"] for e in events if e["type"] == "step_complete"][:3] == ["trip_analyst", "destination_agent", "mobility_agent"]
    assert all("tools" in e["meta"] and "algorithms" in e["meta"] for e in events if e["type"] == "step_complete")


# --- ownership of state --------------------------------------------------------------

def test_an_agent_may_not_write_what_it_does_not_own(make_team):
    team = make_team()
    with pytest.raises(AccessError):
        team.bb.commit(team.agents["destination_agent"], {"route": None})
    team.bb.commit(team.agents["destination_agent"], {"excluded_activity_ids": []})  # its own


def test_an_agent_sees_only_what_it_declared(make_team):
    team = make_team()
    view = team.bb.view(team.agents["mobility_agent"])
    assert set(view) == set(team.agents["mobility_agent"].reads)
    with pytest.raises(AccessError):
        view["draft_itinerary"]
    with pytest.raises(AccessError):
        view.get("final_itinerary")


def test_every_agent_states_its_role_and_peas():
    team = Team()
    assert len(team.agents) == 7  # six planning agents and the edit router
    for agent in team.agents.values():
        assert agent.label and agent.role and agent.architecture
        assert set(agent.peas) == {"performance", "environment", "actuators", "sensors"} and all(agent.peas.values())
        assert agent.reads and agent.writes


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_agents_stay_inside_their_declared_reads_and_writes_across_a_whole_session(strategy):
    """Plans, events and every kind of edit, with strict read/write enforcement: any slip raises AccessError."""
    for text in (NATURE_TRIP, RAINY_TRIP, "6 days in Kochi and Varkala for 2 from 2027-03-10, ₹80,000, culture, by train", "4 days in Delhi"):
        team = Team(strategy, 7, "high")
        team.run_to_end(team.plan(text))
        for message in ("make day 2 lighter", "add Alleppey", "cheapest hotels", "how much does it cost?", "make it 6 days", "Munnar is closed", "strike in Kochi", "blah"):
            team.run_to_end(team.edit(message))
        team.run_to_end(team.disrupt(Disruption(type=DisruptionType.BUDGET_CUT, target="trip", description="cut", new_budget_inr=30000)))


# --- the field ---------------------------------------------------------------------------

def test_the_field_is_reproducible_and_seed_dependent():
    a, b, c = FieldWorld(5), FieldWorld(5), FieldWorld(6)
    days = [date(2027, 7, d).isoformat() for d in range(1, 29)]
    facts = lambda w: [w.heavy_rain("Kerala, India", "Munnar", d) for d in days] + [w.sight_closed("munnar_tea_museum", d) for d in days]
    assert facts(a) == facts(b) and facts(a) != facts(c)


def test_a_switched_off_field_never_surprises():
    world = FieldWorld(1, "off")
    assert not world.enabled
    assert not any(world.heavy_rain("Kerala, India", "Munnar", "2027-07-10") or world.sight_closed("x", "2027-07-10") or world.strike("Kochi", "2027-07-10") for _ in range(3))


def test_the_field_frequencies_match_the_rates():
    july, jan = "2027-07-10", "2027-01-10"
    draws = [FieldWorld(seed) for seed in range(4000)]
    rain = lambda day: sum(w.heavy_rain("Kerala, India", "Kochi", day) for w in draws) / len(draws)
    assert 0.35 < rain(july) < 0.50  # 85% chance of rain in July x half of those days are heavy
    assert rain(jan) < 0.10
    assert 0.03 < sum(w.sight_closed("kochi_fort", jan) for w in draws) / len(draws) < 0.07
    assert 0.02 < sum(w.strike("Kochi", jan) for w in draws) / len(draws) < 0.06


def test_the_team_only_learns_about_what_the_plan_uses():
    team = Team("full", 3, "high")
    team.run_to_end(team.plan(RAINY_TRIP))
    asked = team.bb["observed"].checked
    cities_asked = {fact.split(":")[1].split("|")[0] for fact in asked if fact.startswith("rain:")}
    assert cities_asked <= {d.name for d in team.bb["selected_destinations"]} | {d.name for d in team.bb["candidate_destinations"][:5]}
    assert "Wayanad" not in cities_asked and 0 < team.bb["observed"].checks < 60


def _seed_where(predicate, strategy="full", text=RAINY_TRIP, uncertainty="high", limit=80):
    for seed in range(limit):
        team = Team(strategy, seed, uncertainty)
        team.run_to_end(team.plan(text))
        if predicate(team):
            return seed, team
    raise AssertionError("no seed in range met the condition")


def test_heavy_rain_is_found_and_only_the_architect_replans():
    def rain_replan(team):
        return any(m.type == MsgType.REPLAN and m.summary.startswith("HEAVY_RAIN") for m in team.bus.log)

    seed, team = _seed_where(rain_replan)
    replans = [m for m in team.bus.log if m.type == MsgType.REPLAN]
    assert {m.recipient for m in replans} == {"itinerary_architect"}
    assert team.runs["destination_agent"] == team.runs["mobility_agent"] == team.runs["budget_agent"] == 1
    assert team.runs["itinerary_architect"] > 1
    # what was scheduled now survives the real weather
    assert run_strategy(RAINY_TRIP, "full", seed, "high")["sights_lost"] == 0


def test_the_static_strategy_loses_sights_the_full_system_saves():
    seed, _ = _seed_where(lambda t: run_strategy(RAINY_TRIP, "static", t.seed, "high")["sights_lost"] > 0, strategy="static")
    full, static = run_strategy(RAINY_TRIP, "full", seed, "high"), run_strategy(RAINY_TRIP, "static", seed, "high")
    assert full["sights_lost"] == 0 < static["sights_lost"]
    assert full["value_delivered"] >= static["value_delivered"]
    assert static["field_checks"] == 0 and static["message_counts"].get("REPLAN", 0) == 0


def test_restarting_costs_more_agent_runs_than_a_targeted_replan():
    seed, _ = _seed_where(lambda t: t.runs["itinerary_architect"] > 1)
    full, restart = run_strategy(RAINY_TRIP, "full", seed, "high"), run_strategy(RAINY_TRIP, "restart", seed, "high")
    assert restart["agent_runs"] > full["agent_runs"]
    assert restart["value_delivered"] == pytest.approx(full["value_delivered"], abs=0.5)  # same quality, more work


def test_the_same_seed_gives_the_same_run():
    a, b = Team("full", 11, "high"), Team("full", 11, "high")
    a.run_to_end(a.plan(RAINY_TRIP))
    b.run_to_end(b.plan(RAINY_TRIP))
    assert [(m.sender, m.recipient, m.type, m.summary) for m in a.bus.log] == [(m.sender, m.recipient, m.type, m.summary) for m in b.bus.log]
    assert a.bb["final_itinerary"] == b.bb["final_itinerary"] and a.bb["observed"] == b.bb["observed"]


# --- comparing strategies ----------------------------------------------------------------------

def test_compare_scores_every_strategy_against_the_same_field():
    result = compare_strategies(RAINY_TRIP, 3, "high")
    assert set(result) == set(STRATEGIES)
    assert all(0 <= r["value_ratio"] <= 1.0001 for r in result.values())
    assert result["static"]["field_checks"] == 0 and result["full"]["field_checks"] > 0
    assert result["full"]["value_ratio"] >= result["static"]["value_ratio"]


def test_on_average_the_full_system_beats_planning_once_when_the_field_is_rough():
    ratios = {name: [] for name in ("full", "static")}
    for seed in range(20):
        for name, r in compare_strategies(RAINY_TRIP, seed, "high").items():
            if name in ratios:
                ratios[name].append(r["value_ratio"])
    mean = {k: sum(v) / len(v) for k, v in ratios.items()}
    assert mean["full"] > mean["static"] + 0.03, mean


def test_blackboard_starts_with_every_key_an_agent_may_read():
    board = Blackboard()
    team = Team()
    for agent in team.agents.values():
        assert set(agent.reads) <= set(board.data), agent.name
