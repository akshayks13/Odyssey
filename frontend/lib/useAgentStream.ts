"use client";

import { useCallback, useRef, useState, useTransition } from "react";
import { DisruptPayload, postJson } from "./api";
import { AGENT_ORDER, AgentMeta, AgentStatus, AgentStepEvent, EDIT_AGENT } from "./types";

export interface AgentStreamState {
  statuses: Record<string, AgentStatus>;
  messages: { agent: string; message: string; meta?: AgentMeta }[];
  latest: AgentStepEvent | null;
  isStreaming: boolean;
  error: string | null;
}

const statusesFor = (status: AgentStatus): Record<string, AgentStatus> =>
  Object.fromEntries([...AGENT_ORDER, EDIT_AGENT].map((a) => [a, status])) as Record<string, AgentStatus>;

const initialStatuses = (): Record<string, AgentStatus> => statusesFor("pending");

/**
 * Consume the planning SSE stream and update the agent timeline. Handles a new plan, a prompt edit
 * (only the agents that re-run light up; the rest show as unchanged) and a disruption.
 */
export function useAgentStream() {
  const [state, setState] = useState<AgentStreamState>({
    statuses: initialStatuses(),
    messages: [],
    latest: null,
    isStreaming: false,
    error: null,
  });
  const [, startTransition] = useTransition();
  const abortRef = useRef<AbortController | null>(null);

  const consumeStream = useCallback(async (res: Response) => {
    if (!res.ok || !res.body) {
      throw new Error(`Request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        let event: AgentStepEvent;
        try {
          event = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        startTransition(() => {
          setState((prev) => {
            const next: AgentStreamState = { ...prev, latest: event };

            if (event.type === "step_start" && event.agent) {
              next.statuses = { ...prev.statuses, [event.agent]: "running" };
            } else if (event.type === "step_complete" && event.agent) {
              next.statuses = { ...prev.statuses, [event.agent]: "done" };
              next.messages = [...prev.messages, { agent: event.agent, message: event.message || "", meta: event.meta }];
            } else if (event.type === "error") {
              next.error = event.message || "Unknown error";
              next.isStreaming = false;
              if (event.agent) {
                next.statuses = { ...prev.statuses, [event.agent]: "error" };
              }
            } else if (event.type === "done") {
              next.isStreaming = false;
            }
            return next;
          });
        });
      }
    }
  }, []);

  const connectStream = useCallback(
    async (threadId: string) => {
      const res = await fetch(`/api/stream?threadId=${encodeURIComponent(threadId)}`, {
        method: "GET",
        headers: { Accept: "text/event-stream" },
        cache: "no-store",
        signal: abortRef.current?.signal,
      });
      await consumeStream(res);
    },
    [consumeStream]
  );

  /** Run `queue` (a POST that stores the request), then stream its result. */
  const run = useCallback(
    async (threadId: string, queue: (signal: AbortSignal) => Promise<unknown>, initial: Record<string, AgentStatus>) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState((prev) => ({
        statuses: initial,
        messages: [],
        latest: prev.latest,
        isStreaming: true,
        error: null,
      }));

      try {
        await queue(controller.signal);
        await connectStream(threadId);
      } catch (err) {
        const e = err as Error;
        if (e.name !== "AbortError") {
          setState((prev) => ({ ...prev, error: e.message, isStreaming: false }));
        }
      }
    },
    [connectStream]
  );

  const startPlan = useCallback(
    (message: string, threadId: string) =>
      run(
        threadId,
        (signal) => postJson("/api/plan", { message, thread_id: threadId }, signal),
        { ...initialStatuses(), trip_analyst: "running" }
      ),
    [run]
  );

  /** Change the plan with a sentence. Agents that don't re-run stay "unchanged". */
  const revise = useCallback(
    (threadId: string, message: string) =>
      run(
        threadId,
        (signal) => postJson("/api/revise", { thread_id: threadId, message }, signal),
        { ...statusesFor("kept"), [EDIT_AGENT]: "running" }
      ),
    [run]
  );

  const injectDisruption = useCallback(
    (payload: DisruptPayload) =>
      run(
        payload.thread_id,
        (signal) => postJson("/api/disrupt", payload, signal),
        { ...statusesFor("kept"), critic_replanner: "running" }
      ),
    [run]
  );

  return { ...state, startPlan, revise, injectDisruption };
}
