// Mirrors the Pydantic schemas in backend/models/schemas.py (subset used by the UI).

export interface Coordinates {
  lat: number;
  lng: number;
}

export interface Destination {
  name: string;
  region: string;
  coordinates: Coordinates;
  preference_score: number;
  description: string;
  tags: string[];
  weather_summary?: string | null;
  weather_risk: boolean;
}

export interface RouteLeg {
  origin: string;
  destination: string;
  mode: "road" | "rail" | "air";
  distance_km: number;
  duration_hours: number;
  cost_inr: number;
  available: boolean;
  summary?: string | null;
  airline?: string | null;
  vehicle?: string | null;
  source?: string | null;
  reason?: string | null;
}

export interface Route {
  ordered_destinations: string[];
  legs: RouteLeg[];
  total_distance_km: number;
  total_duration_hours: number;
  total_cost_inr: number;
  search_algorithm: string;
  nodes_expanded: number;
  return_leg?: RouteLeg | null;
}

export interface Hotel {
  name: string;
  destination: string;
  price_per_night_inr: number;
  rating: number;
  source: string;
}

export interface BudgetLineItem {
  category: string;
  amount_inr: number;
  notes: string;
}

export interface BudgetBreakdown {
  hotels_inr: number;
  food_inr: number;
  activities_inr: number;
  transport_inr: number;
  misc_inr: number;
  total_inr: number;
  ceiling_inr: number;
  over_budget_by_inr: number;
  selected_hotels: Hotel[];
  line_items: BudgetLineItem[];
  tradeoff_suggestions: string[];
}

export interface ScheduledItem {
  activity_id: string;
  activity_name: string;
  destination: string;
  start_hour: number;
  end_hour: number;
  category: string;
  kind?: "activity" | "meal";
  cost_inr?: number;
}

export interface DayWeather {
  summary: string;
  condition: string;
  tmin?: number | null;
  tmax?: number | null;
  rain_mm: number;
  rain_chance?: number | null;
  rainy: boolean;
  source: string;
}

export interface ItineraryDay {
  day_number: number;
  destination: string;
  date?: string | null;
  kind?: "sightseeing" | "travel" | "leisure";
  note?: string | null;
  weather?: DayWeather | null;
  items: ScheduledItem[];
  travel_leg?: RouteLeg | null;
  departure_leg?: RouteLeg | null;
  overnight_hotel?: Hotel | null;
}

export interface Itinerary {
  days: ItineraryDay[];
  total_cost_inr: number;
  optimization_score: number;
  score_breakdown: Record<string, number>;
}

export interface ValidationIssue {
  type: string;
  day?: number | null;
  severity: "low" | "medium" | "high";
  message: string;
  target_agent?: string | null;
}

export type AgentStatus = "pending" | "running" | "done" | "error" | "kept";

export interface TripInfo {
  duration_days: number;
  start_date?: string | null;
  travellers: number;
}


/** How an agent reached its answer: shown in the timeline as evidence of tool calling. */
export interface AgentMeta {
  tools?: string[];
  engine?: string | null;
  algorithms?: string[];
  note?: string;
}

export interface AgentStepEvent {
  type: "init" | "step_start" | "step_complete" | "tool_result" | "message" | "custom" | "done" | "error";
  thread_id?: string;
  agent?: string;
  tool?: string;
  message?: string;
  meta?: AgentMeta;
  data?: unknown;
  itinerary?: Itinerary | null;
  budget?: BudgetBreakdown | null;
  route?: Route | null;
  valid?: boolean;
  issues?: ValidationIssue[];
  score?: number | null;
  iteration_count?: number;
  selected_destinations?: Destination[];
  trip?: TripInfo | null;
  assumptions?: string[];
  reply?: string | null;
  summary?: string | null;
  reran_from?: string | null;
}

export const EDIT_AGENT = "edit_router";

export const AGENT_ORDER = [
  "trip_analyst",
  "destination_agent",
  "mobility_agent",
  "budget_agent",
  "itinerary_architect",
  "critic_replanner",
] as const;

export const STEP_LABELS: Record<string, string> = {
  edit_router: "Your change",
  trip_analyst: "Your request",
  destination_agent: "Destinations",
  mobility_agent: "Routes",
  budget_agent: "Budget",
  itinerary_architect: "Schedule",
  critic_replanner: "Review",
};
