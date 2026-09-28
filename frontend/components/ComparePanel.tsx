"use client";

import { useState } from "react";
import { compareStrategies } from "@/lib/api";
import { CompareResponse } from "@/lib/types";
import { cn } from "@/lib/cn";

const WHAT: Record<string, string> = {
  full: "A* order, CSP timetable, targeted replanning, field checks",
  greedy_order: "nearest-neighbour order instead of A*",
  greedy_schedule: "first-fit timetable instead of the CSP",
  restart: "every issue re-runs all agents from the start",
  static: "plan once; no field checks, no replanning",
};

const FIELD: Record<string, string> = { off: "a calm world", normal: "a normal field", high: "a rough field", closures: "many closures", strikes: "many strikes" };

/**
 * Runs the same trip through every strategy, in the same field, and scores each against what really happened
 * (rained-out sights, closed sights, strike days). That is the evidence that each part of the system earns its place.
 */
export function ComparePanel({ threadId, disabled }: { threadId: string; disabled: boolean }) {
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await compareStrategies(threadId));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const rows = result ? Object.entries(result.strategies) : [];
  const best = Math.max(...rows.map(([, r]) => r.value_ratio), 0);
  const fewest = Math.min(...rows.filter(([, r]) => r.field_checks > 0 || r.iterations > 0).map(([, r]) => r.agent_runs), Infinity);

  return (
    <div className="rounded-3xl border border-line bg-white p-5">
      <h3 className="font-semibold text-ink">Does each part earn its place?</h3>
      <p className="mb-4 mt-1 text-sm text-muted">
        Plan this same trip five ways and score every plan against what really happens in the field: rained-out sights, closed sights and strike days.
      </p>
      <button
        type="button"
        onClick={run}
        disabled={disabled || busy}
        className="rounded-full bg-brand-primary px-5 py-2 text-sm font-medium text-white hover:bg-[#164740] disabled:opacity-50"
      >
        {busy ? "Running the strategies…" : result ? "Run again" : "Compare strategies on this trip"}
      </button>
      {error && <p className="mt-3 text-sm text-brand-clay">{error}</p>}

      {result && (
        <div className="mt-5">
          <p className="mb-3 text-xs text-muted">
            Same request, {FIELD[result.uncertainty] ?? result.uncertainty} (seed {result.seed})
            {result.reported ? `, and the ${result.reported} event(s) you reported` : ""}. “Value” is the share of the no-surprise plan’s value that really works.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] text-left text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide text-muted">
                  <th className="pb-2 pr-3 font-medium">Strategy</th>
                  <th className="pb-2 pr-3 font-medium">Value delivered</th>
                  <th className="pb-2 pr-3 font-medium">Sights lost</th>
                  <th className="pb-2 pr-3 font-medium">Agent runs</th>
                  <th className="pb-2 font-medium">Field checks</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([key, r]) => (
                  <tr key={key} className={cn("border-t border-line align-top", key === "full" && "bg-selected/50")}>
                    <td className="py-2 pr-3">
                      <p className="font-medium text-ink">{r.label}</p>
                      <p className="text-xs text-muted">{WHAT[key]}</p>
                    </td>
                    <td className="py-2 pr-3">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-24 overflow-hidden rounded-full bg-wash">
                          <div className={cn("h-full rounded-full", r.value_ratio >= best - 0.005 ? "bg-brand-primary" : "bg-grey-400")} style={{ width: `${Math.min(100, r.value_ratio * 100)}%` }} />
                        </div>
                        <span className="tabular-nums text-ink">{Math.round(r.value_ratio * 100)}%</span>
                      </div>
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-ink">{r.sights_lost}<span className="text-muted"> of {r.sights_planned}</span></td>
                    <td className={cn("py-2 pr-3 tabular-nums", r.agent_runs === fewest && key !== "static" ? "text-brand-sage" : "text-ink")}>{r.agent_runs}</td>
                    <td className="py-2 tabular-nums text-ink">{r.field_checks}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
