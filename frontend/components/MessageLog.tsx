"use client";

import { useEffect, useRef } from "react";
import { BusMessage, STEP_LABELS } from "@/lib/types";
import { cn } from "@/lib/cn";

/** Who is who in the log. The traveller and the environment are not agents, so they are named plainly. */
const NAMES: Record<string, string> = {
  ...STEP_LABELS,
  traveller: "You",
  environment: "Field",
  trip_analyst: "Analyst",
  destination_agent: "Destination",
  mobility_agent: "Mobility",
  budget_agent: "Budget",
  itinerary_architect: "Architect",
  critic_replanner: "Critic",
  edit_router: "Edit router",
};

/** Messages that send work back are the interesting ones, so they stand out. */
const KIND_STYLE: Record<string, string> = {
  REPLAN: "bg-[#f8eae6] text-brand-clay",
  RERUN: "bg-[#f8eae6] text-brand-clay",
  DISRUPTION: "bg-[#f8eae6] text-brand-clay",
  FIELD_REPORT: "bg-[#e6eef2] text-[#2f5d75]",
  ACCEPT: "bg-selected text-brand-sage",
  ANSWER: "bg-selected text-brand-sage",
  NEED_INFO: "bg-selected text-brand-sage",
  BEST_EFFORT: "bg-wash text-grey-500",
};

const name = (id: string) => NAMES[id] ?? id;

/**
 * The messages the agents send each other. Agents never call one another: every arrow in the system is one of these,
 * so this log is the whole conversation.
 */
export function MessageLog({ messages, live }: { messages: BusMessage[]; live: boolean }) {
  const end = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (live) end.current?.scrollIntoView({ block: "nearest" });
  }, [messages.length, live]);

  if (!messages.length) return null;
  return (
    <div className="mt-6">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">
        Messages between agents <span className="ml-1 font-normal normal-case tracking-normal text-grey-500">({messages.length})</span>
      </h2>
      <ol className="max-h-[26rem] space-y-1.5 overflow-y-auto rounded-2xl border border-line bg-white p-3">
        {messages.map((m) => (
          <li key={m.id} className="text-xs leading-snug">
            <p className="text-grey-500">
              <span className="font-medium text-ink">{name(m.from)}</span> → <span className="font-medium text-ink">{name(m.to)}</span>
            </p>
            <p className="mt-0.5 flex flex-wrap items-baseline gap-1.5">
              <span className={cn("rounded-full px-1.5 py-0.5 font-mono text-[10px]", KIND_STYLE[m.kind] ?? "bg-selected text-brand-primary")}>{m.kind}</span>
              <span className="text-muted">{m.summary}</span>
            </p>
          </li>
        ))}
        <div ref={end} />
      </ol>
    </div>
  );
}
