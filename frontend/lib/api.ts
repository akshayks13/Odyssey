export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface DisruptPayload {
  thread_id: string;
  type: "weather" | "closure" | "transport" | "budget_cut";
  target: string;
  description: string;
  day?: number;
  new_budget_inr?: number;
}

export async function fetchItinerary(threadId: string) {
  const res = await fetch(`${API_BASE_URL}/api/itinerary/${threadId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch itinerary (${res.status})`);
  return res.json();
}
