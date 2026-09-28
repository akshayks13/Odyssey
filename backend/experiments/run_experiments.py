"""Compare the four strategies on every scenario, seed by seed.

    cd backend
    python -m experiments.run_experiments --quick          # 3 seeds per scenario, a few seconds
    python -m experiments.run_experiments                  # 30 seeds per scenario (1,350 runs), about a minute
    python -m experiments.run_experiments --replot         # rebuild the tables and charts from results/results.csv

For every (scenario, seed) the four strategies plan the same trip, meet the same field and hear about the same
reported events, so their scores can be compared as pairs. `value_ratio` is the share of the trip's value that
survives contact with the field: value of the sights that really work / value of the plan the same system would
make in a world where nothing goes wrong.

Writes to results/: results.csv (one row per run), summary.csv (mean and std per scenario and strategy),
paired.csv (full system minus each baseline on the same seed, with a 95% interval) and the charts.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.compare import run_strategy  # noqa: E402
from core.strategies import STRATEGIES  # noqa: E402
from experiments.scenarios import BY_NAME, SCENARIOS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUN_COLUMNS = ["scenario", "seed", "strategy", "planned", "valid", "value_planned", "value_delivered", "value_ratio", "sights_planned", "sights_done",
               "sights_lost", "lost_travel_days", "over_budget_pct", "travel_hours", "score", "agent_runs", "messages", "field_checks", "iterations", "wall_ms"]
METRICS = ["value_ratio", "sights_lost", "lost_travel_days", "travel_hours", "agent_runs", "messages", "field_checks", "over_budget_pct", "wall_ms"]
# For a paired comparison "gain" is positive when the full system is better: more value, fewer of the others.
GAIN_SIGN = {"value_ratio": +1, "sights_lost": -1, "agent_runs": -1, "travel_hours": -1}
BASELINES = [name for name in STRATEGIES if name != "full"]


def run_cell(job: tuple[str, int]) -> list[dict]:
    """One (scenario, seed): the reference plan, then every strategy on the same trip and field."""
    scenario_name, seed = job
    scenario = BY_NAME[scenario_name]
    from experiments.scenarios import make_trip

    text, events = make_trip(scenario, seed)
    ideal = run_strategy(text, "full", seed, "off", events)["value_planned"]
    rows = []
    for name in STRATEGIES:
        result = run_strategy(text, name, seed, scenario.uncertainty, events)
        result.update(scenario=scenario_name, seed=seed, strategy=name, value_ratio=round(result["value_delivered"] / ideal, 4) if ideal > 0 else 1.0)
        rows.append({k: result.get(k) for k in RUN_COLUMNS})
    return rows


def run_all(scenarios: list[str], runs: int, workers: int) -> list[dict]:
    jobs = [(name, seed) for name in scenarios for seed in range(runs)]
    if workers <= 1:
        cells = [run_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cells = list(pool.map(run_cell, jobs, chunksize=4))
    return [row for cell in cells for row in cell]


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        for strategy in STRATEGIES:
            mine = [r for r in rows if r["scenario"] == scenario and r["strategy"] == strategy]
            if not mine:
                continue
            row = {"scenario": scenario, "strategy": strategy, "runs": len(mine)}
            for metric in METRICS:
                mean, sd = _mean_sd([float(r[metric]) for r in mine])
                row[f"{metric}_mean"], row[f"{metric}_sd"] = round(mean, 4), round(sd, 4)
            row["valid_pct"] = round(100 * sum(1 for r in mine if r["valid"]) / len(mine), 1)
            out.append(row)
    return out


def paired(rows: list[dict]) -> list[dict]:
    """Full system minus each baseline on the same seed. Positive = the full system is better."""
    index = {(r["scenario"], r["seed"], r["strategy"]): r for r in rows}
    out = []
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        seeds = sorted({r["seed"] for r in rows if r["scenario"] == scenario})
        for baseline in BASELINES:
            for metric, sign in GAIN_SIGN.items():
                diffs = [sign * (float(index[(scenario, s, "full")][metric]) - float(index[(scenario, s, baseline)][metric])) for s in seeds]
                mean, sd = _mean_sd(diffs)
                out.append({
                    "scenario": scenario, "baseline": baseline, "metric": metric, "n": len(diffs),
                    "gain_mean": round(mean, 4), "gain_sd": round(sd, 4), "ci95": round(1.96 * sd / math.sqrt(len(diffs)), 4),
                    "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs),
                })
    return out


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:  # numbers back from text
        for key, value in row.items():
            if key in ("scenario", "strategy"):
                continue
            try:
                row[key] = int(value) if key in ("seed", "sights_planned", "sights_done", "sights_lost", "lost_travel_days", "agent_runs", "messages", "field_checks", "iterations") else float(value) if "." in value or "e" in value.lower() else value
            except ValueError:
                pass
        row["valid"] = row.get("valid") in ("True", True)
    return rows


# --- charts ------------------------------------------------------------------------------------

COLORS = {"full": "#1e5c55", "greedy_order": "#d98a3d", "greedy_schedule": "#b85c38", "restart": "#8a7fb8", "static": "#9a9a9a"}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed (pip install -r requirements-dev.txt): tables were written, charts skipped")
        return None
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def _legend_below(ax) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)


def plot(out: Path, rows: list[dict], summary: list[dict], gains: list[dict]) -> None:
    plt = _pyplot()
    if plt is None:
        return
    by = {(s["scenario"], s["strategy"]): s for s in summary}
    scenarios = list(dict.fromkeys(s["scenario"] for s in summary))
    main = [s for s in scenarios if not BY_NAME[s].group]
    strategies = list(STRATEGIES)

    def ci(scenario: str, strategy: str, metric: str) -> float:
        s = by[(scenario, strategy)]
        return 1.96 * s[f"{metric}_sd"] / math.sqrt(s["runs"])

    def grouped(ax, names, metric, scale=1.0, ylabel=""):
        width = 0.8 / len(strategies)
        for i, strategy in enumerate(strategies):
            xs = [j + i * width - 0.4 + width / 2 for j in range(len(names))]
            ax.bar(xs, [scale * by[(n, strategy)][f"{metric}_mean"] for n in names], width, yerr=[scale * ci(n, strategy, metric) for n in names],
                   color=COLORS[strategy], label=STRATEGIES[strategy].label, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel(ylabel)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    grouped(ax, main, "value_ratio", 100, "% of the no-surprise trip's value delivered")
    ax.set_title("How much of the trip survives the field (mean, 95% interval)")
    _legend_below(ax)
    fig.tight_layout(); fig.savefig(out / "value_by_scenario.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    grouped(ax, main, "agent_runs", 1, "agent runs per trip")
    ax.set_title("Work done: targeted replanning vs restarting everything")
    _legend_below(ax)
    fig.tight_layout(); fig.savefig(out / "agent_runs.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(BASELINES)
    names = list(dict.fromkeys(g["scenario"] for g in gains))
    for i, baseline in enumerate(BASELINES):
        picks = {g["scenario"]: g for g in gains if g["baseline"] == baseline and g["metric"] == "value_ratio"}
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(names))]
        ax.bar(xs, [100 * picks[n]["gain_mean"] for n in names], width, yerr=[100 * picks[n]["ci95"] for n in names], color=COLORS[baseline],
               label=f"full − {STRATEGIES[baseline].label}", error_kw={"lw": 0.8, "capsize": 2})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_ylabel("percentage points of trip value")
    ax.set_title("Paired gain of the full system over each baseline (same trip, same field; 95% interval)")
    _legend_below(ax)
    fig.tight_layout(); fig.savefig(out / "paired_gains.png", dpi=150); plt.close(fig)

    scale = [s for s in scenarios if BY_NAME[s].group == "scaling"]
    if scale:
        days = [BY_NAME[s].days for s in scale]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for strategy in strategies:
            axes[0].plot(days, [100 * by[(s, strategy)]["value_ratio_mean"] for s in scale], "o-", color=COLORS[strategy], label=STRATEGIES[strategy].label)
            axes[1].plot(days, [by[(s, strategy)]["agent_runs_mean"] for s in scale], "o-", color=COLORS[strategy])
            axes[2].plot(days, [by[(s, strategy)]["travel_hours_mean"] for s in scale], "o-", color=COLORS[strategy])
        axes[0].set_ylabel("% of trip value delivered"); axes[1].set_ylabel("agent runs per trip"); axes[2].set_ylabel("hours of travel between cities")
        for ax in axes:
            ax.set_xlabel("trip length (days)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("Scaling with trip length")
        fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(out / "scaling.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    grouped(ax, main, "sights_lost", 1, "sights lost per trip")
    ax.set_title("Sights planned that turn out to be rained out, closed, or on a strike day")
    _legend_below(ax)
    fig.tight_layout(); fig.savefig(out / "sights_lost.png", dpi=150); plt.close(fig)


def print_tables(summary: list[dict], gains: list[dict]) -> None:
    print("\nMean % of the no-surprise trip's value delivered (± std across seeds):")
    print(f"{'scenario':12}" + "".join(f"{STRATEGIES[s].label:>24}" for s in STRATEGIES))
    by = {(r["scenario"], r["strategy"]): r for r in summary}
    for scenario in dict.fromkeys(r["scenario"] for r in summary):
        print(f"{scenario:12}" + "".join(f"{100 * by[(scenario, s)]['value_ratio_mean']:>15.1f} ± {100 * by[(scenario, s)]['value_ratio_sd']:<5.1f}" for s in STRATEGIES))
    print("\nPaired gain of the full system (positive = full is better; ± is the 95% interval; wins/ties/losses out of n seeds):")
    for metric, label in (("value_ratio", "value delivered, points"), ("agent_runs", "agent runs saved"), ("sights_lost", "sights saved"), ("travel_hours", "hours of travel between cities saved")):
        print(f"\n  {label}")
        for g in (g for g in gains if g["metric"] == metric):
            scale = 100 if metric == "value_ratio" else 1
            print(f"  {g['scenario']:12} vs {g['baseline']:8} {scale * g['gain_mean']:+7.2f} ± {scale * g['ci95']:<6.2f} {g['wins']:>3}/{g['ties']:>2}/{g['losses']:<3} of {g['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=30, help="seeds per scenario")
    parser.add_argument("--quick", action="store_true", help="3 seeds per scenario")
    parser.add_argument("--scenarios", default="all", help="comma-separated names, or all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--replot", action="store_true", help="rebuild summary, paired tables and charts from results.csv")
    args = parser.parse_args()

    if args.replot:
        rows = read_csv(args.out / "results.csv")
    else:
        names = [s.name for s in SCENARIOS] if args.scenarios == "all" else args.scenarios.split(",")
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            parser.error(f"unknown scenario(s) {unknown}; choose from {sorted(BY_NAME)}")
        runs = 3 if args.quick else args.runs
        print(f"{len(names)} scenarios x {len(STRATEGIES)} strategies x {runs} seeds = {len(names) * len(STRATEGIES) * runs} runs")
        rows = run_all(names, runs, args.workers)
        write_csv(args.out / "results.csv", rows, RUN_COLUMNS)

    summary, gains = summarize(rows), paired(rows)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "paired.csv", gains)
    print_tables(summary, gains)
    plot(args.out, rows, summary, gains)
    print(f"\nWrote results.csv, summary.csv, paired.csv and charts to {args.out}")


if __name__ == "__main__":
    main()
