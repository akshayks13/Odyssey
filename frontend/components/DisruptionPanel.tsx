"use client";

import { useState } from "react";
import { CloudRain, Landmark, TrainFront, Wallet } from "lucide-react";
import { DisruptPayload } from "@/lib/api";
import { Destination } from "@/lib/types";

export function DisruptionPanel({
  threadId,
  destinations,
  budgetCeiling,
  onInject,
  disabled,
}: {
  threadId: string;
  destinations: Destination[];
  budgetCeiling: number;
  onInject: (payload: DisruptPayload) => void;
  disabled: boolean;
}) {
  const [target, setTarget] = useState("");

  const names = destinations.map((d) => d.name);
  // A closure or edit can remove the city that was selected. Fall back to the first city still in the
  // plan, otherwise the buttons keep saying "Close <a city that is no longer in the trip>".
  const primary = names.includes(target) ? target : names[0] || "";

  const buttonClass =
    "inline-flex items-center gap-1.5 rounded-full border border-line bg-white px-3 py-1.5 text-sm font-medium text-ink hover:bg-wash disabled:opacity-50";

  return (
    <div className="rounded-3xl border border-line bg-white p-5">
      <h3 className="font-semibold text-ink">What if something changes?</h3>
      <p className="mb-4 mt-1 text-sm text-muted">
        Try a closure, weather alert, transport issue, or a tighter budget. We’ll update the itinerary.
      </p>

      <div className="mb-4">
        <label className="mb-1 block text-xs font-medium text-muted">Place</label>
        <select
          value={primary}
          onChange={(e) => setTarget(e.target.value)}
          className="w-full rounded-full border border-line bg-white px-3 py-2 text-sm text-ink focus:border-brand-primary focus:outline-none"
        >
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          disabled={disabled || !primary}
          onClick={() =>
            onInject({
              thread_id: threadId,
              type: "closure",
              target: primary,
              description: `${primary} closed due to landslide/maintenance`,
              day: 1,
            })
          }
          className={buttonClass}
        >
          <Landmark className="h-4 w-4 text-brand-clay" />
          Close {primary}
        </button>
        <button
          disabled={disabled || !primary}
          onClick={() =>
            onInject({
              thread_id: threadId,
              type: "weather",
              target: primary,
              description: `Heavy rain warning issued for ${primary}`,
              day: 1,
            })
          }
          className={buttonClass}
        >
          <CloudRain className="h-4 w-4 text-brand-primary" />
          Weather alert
        </button>
        <button
          disabled={disabled || !primary}
          onClick={() =>
            onInject({
              thread_id: threadId,
              type: "transport",
              target: primary,
              description: `Transport strike affecting routes to ${primary}`,
              day: 1,
            })
          }
          className={buttonClass}
        >
          <TrainFront className="h-4 w-4 text-brand-accent" />
          Transport issue
        </button>
        <button
          disabled={disabled}
          onClick={() =>
            onInject({
              thread_id: threadId,
              type: "budget_cut",
              target: "trip",
              description: "Sponsor withdrew funding — budget reduced",
              new_budget_inr: Math.round(budgetCeiling * 0.7),
            })
          }
          className={buttonClass}
        >
          <Wallet className="h-4 w-4 text-brand-sage" />
          Cut budget 30%
        </button>
      </div>
    </div>
  );
}
