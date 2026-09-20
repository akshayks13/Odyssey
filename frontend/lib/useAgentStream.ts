"use client";

import { useCallback, useRef, useState, useTransition } from "react";
import { API_BASE_URL, DisruptPayload } from "./api";
import { AGENT_ORDER, AgentStatus, AgentStepEvent } from "./types";

export interface AgentStreamState {
  statuses: Record<string, AgentStatus>;
  messages: { agent: string; message: string }[];
  latest: AgentStepEvent | null;
  isStreaming: boolean;
  error: string | null;
}

const initialStatuses = (): Record<string, AgentStatus> =>
  Object.fromEntries(AGENT_ORDER.map((a) => [a, "pending" as AgentStatus])) as Record<string, AgentStatus>;

/**
 * Consume the planning SSE stream and update the agent timeline.
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
              next.messages = [...prev.messages, { agent: event.agent, message: event.message || "" }];
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

  const startPlan = useCallback(
    async (message: string, threadId?: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({
        statuses: { ...initialStatuses(), trip_analyst: "running" },
        messages: [],
        latest: null,
        isStreaming: true,
        error: null,
      });

      try {
        const created = await fetch(`${API_BASE_URL}/api/plan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
          signal: controller.signal,
        });
        if (!created.ok) throw new Error(`Plan request failed (${created.status})`);
        const { thread_id } = await created.json();
        await connectStream(thread_id);
      } catch (err) {
        const e = err as Error;
        if (e.name !== "AbortError") {
          setState((prev) => ({ ...prev, error: e.message, isStreaming: false }));
        }
      }
    },
    [connectStream]
  );

  const injectDisruption = useCallback(
    async (payload: DisruptPayload) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState((prev) => ({
        ...prev,
        isStreaming: true,
        error: null,
        statuses: { ...prev.statuses, critic_replanner: "running" },
      }));

      try {
        const injected = await fetch(`${API_BASE_URL}/api/disrupt`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!injected.ok) throw new Error(`Disruption failed (${injected.status})`);
        await connectStream(payload.thread_id);
      } catch (err) {
        const e = err as Error;
        if (e.name !== "AbortError") {
          setState((prev) => ({ ...prev, error: e.message, isStreaming: false }));
        }
      }
    },
    [connectStream]
  );

  return { ...state, startPlan, injectDisruption };
}
