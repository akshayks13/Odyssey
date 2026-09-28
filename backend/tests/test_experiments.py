"""The experiment harness: same trips for every strategy, paired statistics, reproducible output."""
from __future__ import annotations

import re

import pytest

from core.compare import run_strategy
from core.strategies import STRATEGIES
from experiments.run_experiments import RUN_COLUMNS, paired, read_csv, run_all, summarize, write_csv
from experiments.scenarios import BY_NAME, SCENARIOS, make_trip
from experiments.search_comparison import compare


def test_a_scenario_makes_the_same_trip_for_the_same_seed_and_different_trips_across_seeds():
    scenario = BY_NAME["monsoon"]
    assert make_trip(scenario, 4) == make_trip(scenario, 4)
    assert len({make_trip(scenario, seed)[0] for seed in range(10)}) > 5
    assert "2027-07-10" in make_trip(scenario, 0)[0]


def test_the_budget_cut_scenario_reports_a_30_percent_cut_after_planning():
    text, events = make_trip(BY_NAME["budget_cut"], 2)
    budget = int(re.search(r"₹([\d,]+)", text).group(1).replace(",", ""))
    assert len(events) == 1 and events[0].type.value == "budget_cut"
    assert events[0].new_budget_inr == pytest.approx(0.7 * budget, abs=500)
    assert make_trip(BY_NAME["calm"], 2)[1] == []


def test_every_scenario_is_named_once_and_has_a_purpose():
    assert len({s.name for s in SCENARIOS}) == len(SCENARIOS) >= 6
    assert all(s.what for s in SCENARIOS)


@pytest.fixture(scope="module")
def rows():
    return run_all(["calm", "closures"], runs=3, workers=1)


def test_every_seed_is_run_by_every_strategy(rows):
    assert len(rows) == 2 * 3 * len(STRATEGIES)
    for scenario in ("calm", "closures"):
        for seed in range(3):
            assert {r["strategy"] for r in rows if r["scenario"] == scenario and r["seed"] == seed} == set(STRATEGIES)
    assert all(set(r) == set(RUN_COLUMNS) for r in rows)


def test_in_a_world_where_nothing_goes_wrong_static_and_full_plan_the_same_trip():
    text, events = make_trip(BY_NAME["calm"], 1)
    full, static = run_strategy(text, "full", 1, "off", events), run_strategy(text, "static", 1, "off", events)
    assert full["value_planned"] == static["value_planned"] > 0 and full["field_checks"] == static["field_checks"] == 0


def test_results_are_reproducible(rows):
    again = run_all(["calm", "closures"], runs=3, workers=1)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "wall_ms"} for r in rs]
    assert strip(again) == strip(rows)


def test_summary_and_paired_tables(rows):
    summary, gains = summarize(rows), paired(rows)
    assert len(summary) == 2 * len(STRATEGIES) and all(0 <= s["value_ratio_mean"] <= 1.2 for s in summary)
    baselines = set(STRATEGIES) - {"full"}
    assert {(g["scenario"], g["baseline"], g["metric"]) for g in gains} == {(s, b, m) for s in ("calm", "closures") for b in baselines for m in ("value_ratio", "agent_runs", "sights_lost", "travel_hours")}
    closures = next(g for g in gains if g["scenario"] == "closures" and g["baseline"] == "static" and g["metric"] == "value_ratio")
    assert closures["n"] == 3 and closures["wins"] + closures["ties"] + closures["losses"] == 3


def test_restarting_never_costs_fewer_agent_runs_than_a_targeted_replan(rows):
    gains = [g for g in paired(rows) if g["baseline"] == "restart" and g["metric"] == "agent_runs"]
    assert all(g["gain_mean"] >= 0 for g in gains)


def test_csv_round_trip(rows, tmp_path):
    write_csv(tmp_path / "results.csv", rows, RUN_COLUMNS)
    back = read_csv(tmp_path / "results.csv")
    assert len(back) == len(rows) and back[0]["strategy"] == rows[0]["strategy"] and isinstance(back[0]["agent_runs"], int)
    assert summarize(back)[0]["value_ratio_mean"] == pytest.approx(summarize(rows)[0]["value_ratio_mean"], abs=1e-3)


def test_search_comparison_finds_the_same_optimum_with_fewer_nodes():
    table = compare()
    assert all(r["all_optimal"] for r in table)
    big = table[-1]
    assert big["astar_mst_nodes"] < big["astar_min_edge_nodes"] < big["ucs_nodes"]
    assert table[0]["greedy_optimal"] == table[0]["subsets"] and table[-1]["greedy_optimal"] < table[-1]["subsets"]


def test_charts_are_written_when_matplotlib_is_available(rows, tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.run_experiments import plot

    plot(tmp_path, rows, summarize(rows), paired(rows))
    assert {"value_by_scenario.png", "paired_gains.png", "agent_runs.png", "sights_lost.png"} <= {p.name for p in tmp_path.iterdir()}
