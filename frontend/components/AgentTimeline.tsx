"use client";

import { AGENT_ORDER, AgentMeta, AgentStatus, EDIT_AGENT, STEP_LABELS } from "@/lib/types";
import { stepAccent } from "@/lib/palette";
import { cn } from "@/lib/cn";

const STATUS_LABEL: Record<AgentStatus, string> = {
  pending: "Waiting",
  running: "In progress",
  done: "Done",
  error: "Needs attention",
  kept: "Unchanged",
};

/**
 * Drop only the agent's own name, which the row's heading already shows. Everything else the agent
 * said — tools, algorithms, its reasoning — stays, because that is what the run is judged on. The
 * model and the algorithms are no longer repeated in the prose; they arrive as `meta` and become chips.
 */
function cleanMessage(message: string): string {
  return message
    .replace(
      /^(Trip Analyst|Destination Agent|Mobility Agent|Budget Agent|Itinerary Architect|Critic|Edit Router):\s*/i,
      ""
    )
    .trim();
}

function Chips({ meta }: { meta?: AgentMeta }) {
  const tools = meta?.tools ?? [];
  const algorithms = meta?.algorithms ?? [];
  if (!tools.length && !algorithms.length && !meta?.engine) return null;

  // Tools are listed with how many times each was called, so a model that checked three cities shows it.
  const counted = tools.reduce<Record<string, number>>((acc, t) => ({ ...acc, [t]: (acc[t] ?? 0) + 1 }), {});

  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {Object.entries(counted).map(([tool, n]) => (
        <span
          key={tool}
          title="Tool the model chose to call"
          className="rounded-full bg-selected px-2 py-0.5 font-mono text-[11px] text-brand-primary"
        >
          {tool}
          {n > 1 ? ` ×${n}` : ""}
        </span>
      ))}
      {algorithms.map((algorithm) => (
        <span
          key={algorithm}
          title="Algorithm the code ran"
          className="rounded-full border border-line px-2 py-0.5 text-[11px] text-muted"
        >
          {algorithm}
        </span>
      ))}
      {meta?.engine && (
        <span title="Model that answered" className="rounded-full bg-wash px-2 py-0.5 text-[11px] text-grey-500">
          {meta.engine}
        </span>
      )}
    </div>
  );
}

export function AgentTimeline({
  statuses,
  messages,
  finished = false,
}: {
  statuses: Record<string, AgentStatus>;
  messages: { agent: string; message: string; meta?: AgentMeta }[];
  finished?: boolean; // a saved plan opened without running: its steps are done
}) {
  const lastFor = (agent: string) => [...messages].reverse().find((m) => m.agent === agent);

  // The edit step only appears once a change has been asked for.
  const steps = statuses[EDIT_AGENT] && statuses[EDIT_AGENT] !== "pending" ? [EDIT_AGENT, ...AGENT_ORDER] : [...AGENT_ORDER];

  return (
    <div className="space-y-2">
      {steps.map((agent, i) => {
        const known = statuses[agent] ?? "pending";
        const status = finished && known === "pending" ? "done" : known;
        const entry = lastFor(agent);
        const message = entry?.message;
        const accent = stepAccent[agent];
        return (
          <div
            key={agent}
            className={cn(
              "rounded-2xl border bg-white p-3.5 transition-colors",
              status === "running" && "border-brand-primary/30 bg-selected",
              status === "done" && "border-line",
              status === "pending" && "border-line opacity-70",
              status === "kept" && "border-line opacity-60",
              status === "error" && "border-brand-clay/40 bg-[#f8eae6]"
            )}
          >
            <div className="flex items-center gap-3">
              <span
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-white"
                style={{ background: status === "pending" || status === "kept" ? "#bdbdbd" : accent }}
              >
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-ink">{STEP_LABELS[agent] ?? agent}</span>
                  <span
                    className={cn(
                      "text-xs font-medium",
                      status === "running" && "text-brand-primary",
                      status === "done" && "text-brand-sage",
                      status === "pending" && "text-grey-400",
                      status === "kept" && "text-grey-500",
                      status === "error" && "text-brand-clay"
                    )}
                  >
                    {STATUS_LABEL[status]}
                  </span>
                </div>
                {message && (
                  <p className="mt-1 text-sm leading-snug text-muted">{cleanMessage(message)}</p>
                )}
                <Chips meta={entry?.meta} />
                {entry?.meta?.note && (
                  <p className="mt-1 text-xs leading-snug text-grey-500">{entry.meta.note}</p>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
