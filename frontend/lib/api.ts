// Same-origin: next.config.mjs forwards /api/* to the backend, so no CORS and no address to keep in sync.
import { AgentStepEvent } from "./types";

export interface DisruptPayload {
  thread_id: string;
  type: "weather" | "closure" | "transport" | "budget_cut";
  target: string;
  description: string;
  day?: number;
  new_budget_inr?: number;
}

/** A signal that aborts when either `ms` elapses or `existing` (an unmount/manual abort) fires. */
export function withTimeout(ms: number, existing?: AbortSignal): AbortSignal {
  const controller = new AbortController();
  if (existing?.aborted) {
    controller.abort(existing.reason);
    return controller.signal;
  }
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${ms}ms`)), ms);
  const onAbort = () => controller.abort(existing?.reason);
  existing?.addEventListener("abort", onAbort, { once: true });
  controller.signal.addEventListener("abort", () => {
    clearTimeout(timer);
    existing?.removeEventListener("abort", onAbort);
  }, { once: true });
  return controller.signal;
}

export async function fetchItinerary(threadId: string, signal?: AbortSignal): Promise<Omit<AgentStepEvent, "type">> {
  const res = await fetch(`/api/itinerary/${encodeURIComponent(threadId)}`, {
    cache: "no-store",
    signal: withTimeout(15_000, signal),
  });
  if (!res.ok) throw new Error(`Failed to fetch itinerary (${res.status})`);
  return res.json();
}

export async function postJson<T = unknown>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: withTimeout(20_000, signal),
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) detail = String(data.detail);
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}
