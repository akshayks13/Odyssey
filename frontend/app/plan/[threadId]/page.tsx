"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchItinerary } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";
import { AgentStepEvent } from "@/lib/types";
import { AgentTimeline } from "@/components/AgentTimeline";
import { ItineraryView } from "@/components/ItineraryView";
import { MapView } from "@/components/MapView";
import { BudgetChart } from "@/components/BudgetChart";
import { DisruptionPanel } from "@/components/DisruptionPanel";

export default function PlanPage() {
  const params = useParams<{ threadId: string }>();
  const threadId = params.threadId;
  const { statuses, messages, latest, isStreaming, error, startPlan, injectDisruption } = useAgentStream();

  const [planData, setPlanData] = useState<AgentStepEvent | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!threadId) return;
    const key = `odyssey:${threadId}:message`;
    const pendingMessage = sessionStorage.getItem(key);

    if (pendingMessage) {
      sessionStorage.removeItem(key);
      startPlan(pendingMessage, threadId);
    } else {
      fetchItinerary(threadId)
        .then((data) => setPlanData({ type: "done", ...data }))
        .catch((e) => setLoadError(e.message));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  useEffect(() => {
    if (latest?.type === "done") setPlanData(latest);
  }, [latest]);

  const showResults = planData && planData.itinerary;

  return (
    <div className="bg-wash">
      <div className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
          <div>
            <h1 className="font-display text-lg font-semibold text-ink">Your itinerary</h1>
            <p className="text-sm text-muted">
              {isStreaming ? "Building your plan…" : showResults ? "Ready to review" : "Loading"}
            </p>
          </div>
          <Link href="/" className="text-sm font-medium text-brand-primary hover:underline">
            New search
          </Link>
        </div>
      </div>

      <div className="mx-auto max-w-6xl gap-6 px-4 py-8 sm:px-6 lg:grid lg:grid-cols-[300px_1fr]">
        <aside className="mb-6 lg:mb-0">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">Progress</h2>
          <AgentTimeline statuses={statuses} messages={messages} />

          {(error || loadError) && (
            <div className="mt-4 rounded-2xl border border-brand-clay/20 bg-[#f8eae6] p-3 text-sm text-brand-clay">
              {error || loadError}
            </div>
          )}
        </aside>

        <section className="space-y-6">
          {!showResults && !isStreaming && !loadError && (
            <div className="rounded-3xl border border-dashed border-line bg-white p-10 text-center text-muted">
              Waiting to start planning…
            </div>
          )}

          {!showResults && isStreaming && (
            <div className="rounded-3xl border border-line bg-white p-10 text-center">
              <p className="font-medium text-ink">Putting the trip together</p>
              <p className="mt-1 text-sm text-muted">Destinations, route, budget, and schedule.</p>
            </div>
          )}

          {showResults && (
            <>
              {planData?.valid === false && (
                <div className="rounded-2xl border border-brand-accent/30 bg-wash px-4 py-3 text-sm text-ink">
                  This is a best-effort plan — a few constraints could not be fully resolved.{" "}
                  {planData.issues?.map((i) => i.message).join("; ")}
                </div>
              )}

              <MapView destinations={planData!.selected_destinations || []} route={planData!.route || null} />

              <DisruptionPanel
                threadId={threadId}
                destinations={planData!.selected_destinations || []}
                budgetCeiling={planData!.budget?.ceiling_inr ?? 0}
                onInject={injectDisruption}
                disabled={isStreaming}
              />

              {planData!.itinerary && <ItineraryView itinerary={planData!.itinerary} />}
              {planData!.budget && <BudgetChart budget={planData!.budget} />}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
