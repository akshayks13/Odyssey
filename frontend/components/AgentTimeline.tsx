"use client";

import { AGENT_ORDER, AgentStatus, STEP_LABELS } from "@/lib/types";
import { stepAccent } from "@/lib/palette";
import { cn } from "@/lib/cn";

const STATUS_LABEL: Record<AgentStatus, string> = {
  pending: "Waiting",
  running: "In progress",
  done: "Done",
  error: "Needs attention",
};

function cleanMessage(message: string): string {
  return message
    .replace(
      /^(Trip Analyst|Destination Agent|Mobility Agent|Budget Agent|Itinerary Architect|Critic):\s*/i,
      ""
    )
    .replace(/\s+via\s+.+$/i, "")
    .replace(/weighted_astar\s+/i, "")
    .trim();
}

export function AgentTimeline({
  statuses,
  messages,
}: {
  statuses: Record<string, AgentStatus>;
  messages: { agent: string; message: string }[];
}) {
  const lastMessageFor = (agent: string) =>
    [...messages].reverse().find((m) => m.agent === agent)?.message;

  return (
    <div className="space-y-2">
      {AGENT_ORDER.map((agent, i) => {
        const status = statuses[agent] ?? "pending";
        const message = lastMessageFor(agent);
        const accent = stepAccent[agent];
        return (
          <div
            key={agent}
            className={cn(
              "rounded-2xl border bg-white p-3.5 transition-colors",
              status === "running" && "border-brand-primary/30 bg-selected",
              status === "done" && "border-line",
              status === "pending" && "border-line opacity-70",
              status === "error" && "border-brand-clay/40 bg-[#f8eae6]"
            )}
          >
            <div className="flex items-center gap-3">
              <span
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold text-white"
                style={{ background: status === "pending" ? "#bdbdbd" : accent }}
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
                      status === "error" && "text-brand-clay"
                    )}
                  >
                    {STATUS_LABEL[status]}
                  </span>
                </div>
                {message && (
                  <p className="mt-1 text-sm leading-snug text-muted">{cleanMessage(message)}</p>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
