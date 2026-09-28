"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { CloudRain, HelpCircle, Info } from "lucide-react";
import { fetchItinerary } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";
import { AgentStepEvent, BusMessage, Destination } from "@/lib/types";
import { AgentTimeline } from "@/components/AgentTimeline";
import { ChatTurn, EditBox } from "@/components/EditBox";
import { ItineraryView } from "@/components/ItineraryView";
import { MapView } from "@/components/MapView";
import { BudgetChart } from "@/components/BudgetChart";
import { DisruptionPanel } from "@/components/DisruptionPanel";
import { ComparePanel } from "@/components/ComparePanel";
import { MessageLog } from "@/components/MessageLog";

function DestinationStrip({ destinations }: { destinations: Destination[] }) {
  if (!destinations.length) return null;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {destinations.map((d) => (
        <div key={d.name} className="rounded-2xl border border-line bg-white px-3 py-2">
          <p className="text-sm font-medium text-ink">{d.name}</p>
          {d.weather_summary && <p className="text-xs leading-snug text-muted">{d.weather_summary}</p>}
        </div>
      ))}
    </div>
  );
}

export default function PlanPage() {
  const params = useParams<{ threadId: string }>();
  const threadId = params.threadId;
  const { statuses, messages, busMessages, latest, isStreaming, error, startPlan, revise, injectDisruption } = useAgentStream();

  const [planData, setPlanData] = useState<AgentStepEvent | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [busLog, setBusLog] = useState<BusMessage[]>([]);
  const request = useRef("");
  const handled = useRef<AgentStepEvent | null>(null);

  useEffect(() => {
    if (!threadId) return;
    const key = `odyssey:${threadId}:message`;
    const pendingMessage = sessionStorage.getItem(key);

    const planningKey = `odyssey:${threadId}:planning`;
    if (pendingMessage) {
      sessionStorage.removeItem(key);
      sessionStorage.setItem(planningKey, pendingMessage);
      request.current = pendingMessage;
      const options = JSON.parse(sessionStorage.getItem(`odyssey:${threadId}:options`) || "{}");
      startPlan(pendingMessage, threadId, options);
      return;
    }
    if (sessionStorage.getItem(planningKey)) {
      return;
    }
    fetchItinerary(threadId)
      .then((data) => {
        setPlanData({ type: "done", ...data });
        setBusLog(data.messages ?? []);
      })
      .catch((e) => setLoadError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  // One reaction per finished run: a new plan, an edit, a question, or a request for missing info.
  useEffect(() => {
    if (latest?.type !== "done" || handled.current === latest) return;
    handled.current = latest;
    sessionStorage.removeItem(`odyssey:${threadId}:planning`);
    setLoadError(null);
    setBusLog(latest.messages ?? []);

    if (latest.reply) {
      if (planData?.itinerary) setTurns((t) => [...t, { role: "assistant", text: latest.reply as string }]);
      else setQuestion(latest.reply);
    }
    if (latest.itinerary) {
      setPlanData(latest);
      setQuestion(null);
      if (latest.summary) setTurns((t) => [...t, { role: "assistant", text: `${latest.summary}` }]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latest]);

  function handleSend(message: string) {
    setTurns((t) => [...t, { role: "user", text: message }]);
    revise(threadId, message);
  }

  function handleAnswer(event: FormEvent) {
    event.preventDefault();
    if (!answer.trim()) return;
    request.current = `${request.current}. ${answer.trim()}`;
    setAnswer("");
    setQuestion(null);
    sessionStorage.setItem(`odyssey:${threadId}:planning`, request.current);
    startPlan(request.current, threadId, JSON.parse(sessionStorage.getItem(`odyssey:${threadId}:options`) || "{}"));
  }

  const showResults = planData && planData.itinerary;
  const rainyDays = (planData?.itinerary?.days ?? []).filter((d) => d.weather?.rainy);

  return (
    <div className="bg-wash">
      <div className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
          <div>
            <h1 className="font-display text-lg font-semibold text-ink">Your itinerary</h1>
            <p className="text-sm text-muted">
              {isStreaming ? "Working on it…" : showResults ? "Ready to review" : question ? "One quick question" : "Loading"}
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
          <AgentTimeline statuses={statuses} messages={messages} finished={Boolean(showResults) && !isStreaming} />

          {(error || loadError) && (
            <div className="mt-4 rounded-2xl border border-brand-clay/20 bg-[#f8eae6] p-3 text-sm text-brand-clay">
              {error || loadError}
            </div>
          )}

          <MessageLog messages={isStreaming ? [...busLog, ...busMessages] : busLog} live={isStreaming} />
        </aside>

        <section className="space-y-6">
          {question && !isStreaming && (
            <form onSubmit={handleAnswer} className="rounded-3xl border border-line bg-white p-6">
              <p className="flex items-center gap-2 font-semibold text-ink">
                <HelpCircle className="h-5 w-5 text-brand-primary" />
                {question}
              </p>
              <div className="mt-4 flex gap-2">
                <input
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  autoFocus
                  placeholder="e.g. Kerala, in December"
                  className="flex-1 rounded-full border border-line px-4 py-2.5 text-sm focus:border-brand-primary focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={!answer.trim()}
                  className="rounded-full bg-brand-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-[#9A4B2E] disabled:opacity-40"
                >
                  Plan it
                </button>
              </div>
            </form>
          )}

          {!showResults && !question && !isStreaming && !loadError && (
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
              {!!planData?.assumptions?.length && (
                <div className="flex gap-3 rounded-2xl border border-line bg-white px-4 py-3 text-sm text-muted">
                  <Info className="mt-0.5 h-4 w-4 shrink-0 text-brand-primary" />
                  <p>We assumed {planData.assumptions.join(", ")}. Tell us below to change any of these.</p>
                </div>
              )}

              {planData?.valid === false && (
                <div className="rounded-2xl border border-brand-accent/30 bg-wash px-4 py-3 text-sm text-ink">
                  This is a best-effort plan — a few constraints could not be fully resolved.{" "}
                  {planData.issues?.map((i) => i.message).join("; ")}
                </div>
              )}

              {rainyDays.length > 0 && (
                <div className="flex gap-3 rounded-2xl border border-line bg-white px-4 py-3 text-sm text-muted">
                  <CloudRain className="mt-0.5 h-4 w-4 shrink-0 text-brand-primary" />
                  <p>
                    Rain is likely on {rainyDays.map((d) => `day ${d.day_number} (${d.destination})`).join(", ")}
                    {rainyDays.some((d) => d.weather?.source === "field report") ? " — including days the field check reported" : " — typical for that month"}. Indoor
                    sights are scheduled first on those days; pack a raincoat.
                  </p>
                </div>
              )}

              {planData?.stats && (
                <p className="text-xs text-muted">
                  {planData.stats.agent_runs} agent runs · {planData.stats.messages} messages
                  {planData.stats.field_checks > 0 ? ` · ${planData.stats.field_checks} facts checked in the field (weather, closures, strikes) for the days, sights and journeys in this plan` : ""}
                </p>
              )}

              <DestinationStrip destinations={planData!.selected_destinations || []} />

              <MapView destinations={planData!.selected_destinations || []} route={planData!.route || null} />

              <EditBox turns={turns} disabled={isStreaming} onSend={handleSend} />

              <ItineraryView itinerary={planData!.itinerary!} hotels={planData!.budget?.selected_hotels || []} />
              {planData!.budget && <BudgetChart budget={planData!.budget} />}

              <DisruptionPanel
                threadId={threadId}
                destinations={planData!.selected_destinations || []}
                budgetCeiling={planData!.budget?.ceiling_inr ?? 0}
                onInject={injectDisruption}
                disabled={isStreaming}
              />

              <ComparePanel threadId={threadId} disabled={isStreaming} />
            </>
          )}
        </section>
      </div>
    </div>
  );
}
