"use client";

import { useCallback, useRef, useState, useTransition } from "react";
import { DisruptPayload, postJson, withTimeout } from "./api";
import { AGENT_ORDER, AgentMeta, AgentStatus, AgentStepEvent, EDIT_AGENT } from "./types";

// A live run can legitimately take several minutes under provider rate-limiting (a retry
// cascade across every Gemini key plus Groq adds up). This is a backstop against a genuinely
// dead connection, not a normal-latency budget.
const STREAM_TIMEOUT_MS = 10 * 60_000;

export interface AgentStreamState {
  statuses: Record<string, AgentStatus>;
  messages: { agent: string; message: string; meta?: AgentMeta }[];
  /** The tool an agent is calling right now, live from `tool_result` frames — real-time
   * evidence that a step is doing something during a long wait. Cleared once the agent's
   * step completes. */
  liveTool: Record<string, string | null>;
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
    liveTool: {},
    latest: null,
    isStreaming: false,
    error: null,
  });
  const [, startTransition] = useTransition();
  const abortRef = useRef<AbortController | null>(null);

  /** Returns whether a terminal frame (done/error) was actually seen, so the caller can tell
   * a clean finish apart from a connection that just quietly ended (proxy timeout, backend
   * crash) without ever saying so — that used to leave "Working on it…" on screen forever. */
  const consumeStream = useCallback(async (res: Response) => {
    if (!res.ok || !res.body) {
      throw new Error(`Request failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawTerminalFrame = false;

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

        if (event.type === "error" || event.type === "done") sawTerminalFrame = true;

        startTransition(() => {
          setState((prev) => {
            const next: AgentStreamState = { ...prev, latest: event };

            if (event.type === "step_start" && event.agent) {
              next.statuses = { ...prev.statuses, [event.agent]: "running" };
            } else if (event.type === "tool_result" && event.agent) {
              next.liveTool = { ...prev.liveTool, [event.agent]: event.tool || null };
            } else if (event.type === "step_complete" && event.agent) {
              next.statuses = { ...prev.statuses, [event.agent]: "done" };
              next.messages = [...prev.messages, { agent: event.agent, message: event.message || "", meta: event.meta }];
              next.liveTool = { ...prev.liveTool, [event.agent]: null };
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
    return sawTerminalFrame;
  }, []);

  const connectStream = useCallback(
    async (threadId: string) => {
      const res = await fetch(`/api/stream?threadId=${encodeURIComponent(threadId)}`, {
        method: "GET",
        headers: { Accept: "text/event-stream" },
        cache: "no-store",
        signal: withTimeout(STREAM_TIMEOUT_MS, abortRef.current?.signal),
      });
      return consumeStream(res);
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
        liveTool: {},
        latest: prev.latest,
        isStreaming: true,
        error: null,
      }));

      try {
        await queue(controller.signal);
        const sawTerminalFrame = await connectStream(threadId);
        if (!sawTerminalFrame && !controller.signal.aborted) {
          setState((prev) => ({ ...prev, error: "The connection ended before the plan finished. Please try again.", isStreaming: false }));
        }
      } catch (err) {
        const e = err as Error;
        if (e.name !== "AbortError") {
          setState((prev) => ({ ...prev, error: e.message, isStreaming: false }));
        }
      } finally {
        // Belt and braces: whatever happened above, this run must never leave the UI spinning.
        if (!controller.signal.aborted) {
          setState((prev) => (prev.isStreaming ? { ...prev, isStreaming: false } : prev));
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

  /** Reconnect to an already-queued run — no new POST — for a reload or tab reopen while a
   * plan/edit/disruption is still in flight. The backend resumes from its checkpoint (or, after
   * a restart, from the saved state) and replays whatever already happened before continuing. */
  const resumeStream = useCallback(
    (threadId: string) => run(threadId, async () => undefined, initialStatuses()),
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

  return { ...state, startPlan, revise, injectDisruption, resumeStream };
}
