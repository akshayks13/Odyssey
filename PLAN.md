---
name: Odyssey Travel System
overview: Design and implement Odyssey — a 6-agent LangGraph-powered adaptive travel planning system with real API integrations (Mapbox, OpenWeatherMap) plus Gemini-generated hotel/flight market data, weighted A* routing, OR-Tools VRPTW scheduling, and a Next.js frontend with live agent-progress streaming via SSE.
todos:
  - id: repo-init
    content: Initialize git repo, push to GitHub (akshayks13/Odyssey), set up monorepo structure
    status: completed
  - id: state-models
    content: Define TripState TypedDict, all Pydantic schemas (TripSpec, Destination, Activity, Route, Itinerary, ValidationReport, Disruption)
    status: completed
  - id: tools-layer
    content: "Implement all tools: mapbox_api.py (Geocoding/Directions), foursquare_api.py (Places/hours), travel_market.py (Gemini hotels/flights), weather.py (OpenWeatherMap), cost_calculator.py, preference_scorer.py, schedule_validator.py, budget_validator.py"
    status: completed
  - id: algorithms
    content: Implement weighted A* (astar.py — state=(location, visited_set, elapsed_time), haversine heuristic, weight escalation for anytime behavior), VRPTW scheduler (csp_solver.py — OR-Tools pywrapcp.RoutingModel with time windows from opening hours), multi-objective scorer (optimizer.py)
    status: completed
  - id: agents
    content: "Implement all 6 LangChain/LangGraph agents: trip_analyst, destination_agent, budget_agent, mobility_agent, itinerary_architect, critic_replanner"
    status: completed
  - id: langgraph-graph
    content: "Wire LangGraph StateGraph: Analyst→Destination→Mobility→Budget→Architect→Critic; targeted replan edges; max 3 iteration guard; MemorySaver"
    status: completed
  - id: fastapi-backend
    content: FastAPI app with SSE streaming endpoint (POST /api/plan), disruption injection endpoint (POST /api/disrupt), itinerary CRUD, CORS config
    status: completed
  - id: frontend
    content: "Next.js 14 App Router (nodejs runtime): landing page, live planning page with AgentTimeline (SSE via TransformStream proxy), ItineraryView (day accordion), MapView (Mapbox GL JS), DisruptionPanel, BudgetChart (recharts), useAgentStream hook"
    status: completed
  - id: tests
    content: Unit tests for A*, CSP, preference scorer, budget validator; agent-level tests with mocked tools; 3 full scenario integration tests (normal, weather disruption, budget overrun)
    status: completed
  - id: local-setup
    content: .env.example with API keys; README with local run instructions (uvicorn + npm run dev); Kerala seed data fallback when live APIs fail
    status: completed
isProject: false
---

# Odyssey — Multi-Agent Adaptive Travel Planning System

## Architecture Overview

```mermaid
flowchart TD
    User["User natural language"]
    TA["1. Trip Analyst"]
    DA["2. Destination Discovery"]
    MA["3. Mobility A-star"]
    BA["4. Budget Optimization"]
    IA["5. Itinerary Architect"]
    CR["6. Critic and Replanner"]
    FE["Final itinerary"]

    User --> TA
    TA -->|"TripSpec"| DA
    DA -->|"selected cities"| MA
    MA -->|"ordered route plus travel cost"| BA
    BA -->|"hotels plus full budget"| IA
    IA -->|"draft schedule"| CR
    CR -->|"valid"| FE
    CR -->|"destination issue"| DA
    CR -->|"route issue"| MA
    CR -->|"budget issue"| BA
    CR -->|"schedule issue"| IA
```

---

## Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Agent Framework**: LangGraph `StateGraph` (linear specialists + Critic routing) + LangChain
- **LLM**: Gemini Flash-Lite (`GEMINI_MODEL`, default `gemini-3.1-flash-lite`); Groq if Gemini is rate-limited
- **Fallback**: optional Groq model configured through environment variables; use only when the selected Gemini model is rate-limited
- **LangGraph checkpointing**: `MemorySaver` for local/demo; optional `AsyncPostgresSaver` later if persistence is needed
- **Observability**: LangSmith (free tier) — trace every agent step, visualize tool calls, inspect reasoning chains; set via `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`
- **Algorithms**: weighted A* for city order; OR-Tools VRPTW/CP-SAT for day schedules; weighted-sum scorer (no scipy)
- **External APIs**:
  - **Mapbox**: map rendering, geocoding, road directions and travel-time matrices
  - **Foursquare Places**: nearby POIs; LLM fills visitor sights when that list is empty or not useful
  - **LLM travel market**: hotels and airfares from Gemini, then Groq
  - **OpenWeatherMap**: current weather + 5-day / 3-hour forecast
- **Database**: SQLite for saved itineraries (enough for the course demo)
- **Cache**: None
- **Offline fallback**: `data/kerala_seed.json` for cities present in that file when a live lookup fails
- **Frontend**: Next.js 14 App Router (nodejs runtime) + Tailwind + shadcn/ui + Mapbox GL JS
- **Streaming**: Server-Sent Events — FastAPI `StreamingResponse` + LangGraph `astream(version="v2", stream_mode=[...])` → Next.js `TransformStream` route handler
- **Deployment**: local run only (no Docker for v1)

---

## The 6 Agents

Every specialist is a LangGraph node. The model **decides**; tools and algorithms **compute**. With no LLM key (or `ODYSSEY_DISABLE_LLM=1`), the same tools still run on a heuristic path.

Graph order (locked): Analyst → Destination → Mobility → Budget → Architect → Critic.

---

### Agent 1 — Trip Analyst (`backend/agents/trip_analyst.py`)

- **Role**: Parser. Turn the user's sentence into a `TripSpec`. Later agents plan; this one only extracts.
- **Input**: `raw_input` (natural language)
- **Output**: `trip_spec` (`destination_region`, `origin_city`, days, travellers, budget, preference weights, pace, `needs_clarification`)
- **Decides (LLM)**: Which place they asked to visit; home/arrival city if they said "from X to Y"; missing-field defaults
- **Computes**: Regex/keyword heuristic when no LLM is configured (duration, budget, "from X to Y", named city, region keywords)
- **Must not**: Pick sights, hotels, a visit order, or a day plan. `origin_city` is arrival, not a sightseeing stop. Do not default the region to Kerala unless they asked for Kerala / a Kerala town or nothing was named.
- **Tools bound**: `geocode_location` (Mapbox), `validate_trip_schema`
- **Instruction**: SYSTEM_PROMPT in the file — "You are Odyssey's Trip Analyst — the parser, not the planner."

---

### Agent 2 — Destination Discovery (`backend/agents/destination_agent.py`)

- **Role**: Explorer. Choose which cities to keep as overnight/sightseeing stops and load attractions.
- **Input**: `trip_spec` (+ disruptions on replan)
- **Output**: `candidate_destinations`, `selected_destinations`, `candidate_activities`
- **Decides (LLM)**: Short list of city names from the ranking (never a disrupted city; never the origin/home city)
- **Computes**: Weighted cosine preference score; weather risk; `max_destinations = duration_days // 2` (raised if the user named more cities); extra geocode of named towns the region search missed
- **Must not**: Invent a schedule, pick hotels, or choose travel order (Mobility does A*)
- **Tools bound**:
  - `search_destinations` — bundled cities when the region matches that dataset, else Mapbox geocode
  - `search_attractions` — LLM visitor sights first; else Foursquare nearby; else seed for cities in that file
  - `get_weather_forecast` — OpenWeatherMap
  - `score_preference_match` — cosine similarity
  - `get_place_photos` — Foursquare, else Mapbox Static
- **Instruction**: "You are Odyssey's Destination Discovery agent — the explorer, not the scheduler."

---

### Agent 3 — Mobility & Routing (`backend/agents/mobility_agent.py`)

- **Role**: Mover. Order the cities Destination already chose; label each hop road / rail / air.
- **Input**: `selected_destinations`, `trip_spec` (including optional `origin_city`)
- **Output**: `route` (ordered cities, legs, totals, A* stats)
- **Decides (LLM)**: Start city among selected stops; mode label per hop when that quote is available. Honours "by train / flight / road"
- **Computes**: NetworkX `DiGraph` of Mapbox **road** times; weighted A* visit order; then labels. Optional arrival hop `origin_city → first stop`. Single-city trips have no in-region A* hops — only that arrival hop if origin was named.
- **Must not**: Add new cities, invent visit order, or pick hotels
- **Tools bound**: `quote_transport`, `search_flights`, `check_transport_availability`, `calculate_route_cost`
- **Algorithms (not LLM tools)**: `build_travel_graph`, `astar_route_search` (`algorithms/astar.py`)
- **Instruction**: "You are Odyssey's Mobility agent — the mover, not the city picker."

---

### Agent 4 — Budget Optimization (`backend/agents/budget_agent.py`)

- **Role**: Money optimizer. Runs **after** Mobility so transport cost is already in the total.
- **Input**: `trip_spec`, `selected_destinations`, `route`, activities
- **Output**: `budget_breakdown`, `accommodation_options`, `excluded_activity_ids`
- **Decides (LLM)**: Hotel tier (budget/mid/premium); cheapest vs highest-rated hotel; which activities to drop if over ceiling
- **Computes**: Line items (hotels × nights, food, activities, transport); `validate_budget`; cheapest-damage cuts keep ≥1 activity per city
- **Must not**: Reorder cities, invent rupee totals, or build the hour schedule. Hotel/flight quotes are market estimates, not bookable inventory.
- **Tools bound**: `search_hotels`, `search_hotel_offers`, `estimate_food_costs`, `validate_budget`, `generate_tradeoff_options`
- **In-node (not bound)**: `calculate_activity_costs`
- **Instruction**: "You are Odyssey's Budget agent — the money optimizer, not the scheduler."

---

### Agent 5 — Itinerary Architect (`backend/agents/itinerary_architect.py`)

- **Role**: Scheduler. Pack activities, meals, and overnight stays into days on the A* city order.
- **Input**: ordered cities, activities, route, budget (selected hotels)
- **Output**: `draft_itinerary`, `optimization_score`
- **Decides (LLM)**: Pace = relaxed (2 sights, 09–18) / moderate (3, 08–21) / packed (4, 07–22)
- **Computes**: Days per city; opening hours; Mapbox travel matrix; OR-Tools VRPTW; overnight hotel every night except the last; multi-objective score
- **Must not**: Reorder cities, pick hotels, or invent travel times
- **Tools bound**: `get_opening_hours`, `travel_time_matrix`, `validate_time_windows`
- **Algorithms (not LLM tools)**: `solve_day_schedule` (OR-Tools VRPTW), `compute_score`
- **In-node**: `check_schedule_conflicts` after the solve
- **Instruction**: "You are Odyssey's Itinerary Architect — the scheduler, not the router."

---

### Agent 6 — Critic & Replanner (`backend/agents/critic_replanner.py`)

- **Role**: Coordinator. Validate the draft; if invalid, re-invoke **one** specialist (max 3 loops). Never restarts Trip Analyst.
- **Input**: draft itinerary, budget, disruptions, `iteration_count`
- **Output**: `validation_report`, `replan_directives`, `final_itinerary` (when valid or out of loops)
- **Decides (LLM)**: Which of destination / mobility / budget / architect to call for the cheapest-damage repair
- **Computes**: Budget / schedule / weather / closure / transport checks; severity sort. Last allowed pass returns a best-effort plan with warnings.
- **Must not**: Rewrite days itself or fan-out every agent
- **Tools bound**: `validate_budget`, `check_schedule_conflicts`, `check_transport_disruptions`, `check_attraction_availability`, `check_weather_disruptions`
- **Routing**: LangGraph conditional edges (`orchestration/routing.py`). The JSON reply **is** the replan directive — there is no separate `generate_replan_directive` tool.
- **Instruction**: "You are Odyssey's Critic & Replanner — the coordinator, not a seventh planner."

---

## Shared State (LangGraph)

```python
# orchestration/state.py
class TripState(TypedDict):
    # Input
    raw_input: str
    trip_spec: Optional[TripSpec]

    # Discovery
    candidate_destinations: List[Destination]
    candidate_activities: Dict[str, List[Activity]]

    # Optimization
    selected_destinations: List[Destination]
    route: Optional[Route]
    budget_breakdown: Optional[BudgetBreakdown]
    accommodation_options: List[Hotel]

    # Schedule
    draft_itinerary: Optional[Itinerary]
    final_itinerary: Optional[Itinerary]

    # Validation
    validation_report: Optional[ValidationReport]
    replan_directives: List[ReplanDirective]
    iteration_count: int

    # Live events
    disruptions: List[Disruption]
    agent_messages: Annotated[List[str], operator.add]  # streaming log
    conflicts: List[AgentConflict]  # Destination vs Budget vs Mobility disagreements
    optimization_score: float
```

---

## Algorithms

### A* Destination Ordering (`algorithms/astar.py`)
- **Why A* when Maps API exists**: Mapbox/Google gives you road directions between two chosen points (A→B). It does NOT decide the **order** to visit multiple cities (should Day 2 be Munnar or Thekkady?). That ordering problem requires search — that's A*.
- Two distinct layers:
  - **A* (meta-level)**: Finds the optimal order to visit N candidate destinations. Graph nodes = cities, edges = travel time from Mapbox Distance Matrix. State = `(current_city, frozenset(visited), elapsed_days)`. Heuristic = min remaining travel between unvisited cities.
  - **Mapbox Directions API (road-level)**: Once order is decided, gives actual road route, duration, and polyline for the map.
- Weight escalation: `f(n) = g(n) + (1 + ε)·h(n)` — anytime behavior, always returns a plan
- Used by: Mobility Agent

### VRPTW Scheduling (`algorithms/csp_solver.py`)
- Library: `ortools.constraint_solver.pywrapcp.RoutingModel` (Vehicle Routing with Time Windows)
- Nodes = activities + hotels; edges = travel time matrix from Mapbox Directions API
- Time windows per node = attraction opening hours (fetched from Foursquare)
- Service times = activity durations; vehicle capacity = daily hour budget
- Soft constraints: meal windows (penalty for violations), rest blocks
- Objective: Minimize travel time while maximizing preference-weighted activity score
- Used by: Itinerary Architect Agent

### Multi-Objective Scoring (`algorithms/optimizer.py`)
- Formula: `Score = w_p·P + w_q·Q + w_r·R + w_b·B − w_t·T − w_c·C`
- Weights personalized per traveler archetype (adventure / relaxed / budget)
- Used by: Itinerary Architect + Critic agents

---

## LangGraph Graph Topology

```python
# orchestration/graph.py
graph = StateGraph(TripState)
graph.add_node("trip_analyst", trip_analyst_node)
graph.add_node("destination_agent", destination_node)
graph.add_node("budget_agent", budget_node)
graph.add_node("mobility_agent", mobility_node)
graph.add_node("itinerary_architect", itinerary_node)
graph.add_node("critic_replanner", critic_node)

graph.set_entry_point("trip_analyst")
graph.add_edge("trip_analyst", "destination_agent")
graph.add_edge("destination_agent", "mobility_agent")
graph.add_edge("mobility_agent", "budget_agent")
graph.add_edge("budget_agent", "itinerary_architect")
graph.add_edge("itinerary_architect", "critic_replanner")

# Replan must not wait for a parallel join:
# dest issue  → Destination → Mobility → Budget → Architect → Critic
# route issue → Mobility → Budget → Architect → Critic
# budget issue → Budget → Architect → Critic
# schedule issue → Architect → Critic

# Targeted replan routing — max 3 iterations guarded by iteration_count
graph.add_conditional_edges("critic_replanner", route_after_critic, {
    "valid": END,
    "max_iterations": END,       # emit partial plan with warnings
    "replan_destination": "destination_agent",
    "replan_budget": "budget_agent",
    "replan_mobility": "mobility_agent",
    "rebuild_schedule": "itinerary_architect",
})

# Compile with MemorySaver for the course demo
app = graph.compile(checkpointer=MemorySaver())
```

---

## API Layer (FastAPI)

- `POST /api/plan` → starts LangGraph session, returns `thread_id`, begins streaming
- `GET /api/plan/{thread_id}/stream` → SSE stream; uses `graph.astream(version="v2", stream_mode=["messages","updates","custom"], subgraphs=True)`
- `POST /api/disrupt` → injects `Disruption` into live state via `graph.aupdate_state()` → triggers Critic replan
- `GET /api/itinerary/{thread_id}` → fetch finalized itinerary from checkpoint
- `GET /api/health` → readiness check

SSE event types (typed stream):
```json
{"type": "step_start",   "agent": "destination_agent", "message": "Searching destinations..."}
{"type": "tool_result",  "agent": "mobility_agent",    "tool": "astar_route_search", "data": {...}}
{"type": "step_complete","agent": "itinerary_architect","data": {...draft_itinerary...}}
{"type": "done",         "data": {...final_itinerary...}, "score": 0.87}
{"type": "error",        "agent": "budget_agent",       "message": "hotel search timeout, retrying..."}
```

Key FastAPI pattern: graph compiled **once at startup** via `lifespan`; `thread_id = user_id:session_id`; `X-Accel-Buffering: no` header for Nginx.

---

## Frontend (Next.js)

- `app/page.tsx` — Hero + natural language input form
- `app/plan/[threadId]/page.tsx` — Live planning view with agent progress + map
- `app/api/stream/route.ts` — Next.js Route Handler (`runtime = 'nodejs'`); proxies FastAPI SSE via `TransformStream`; forwards `req.signal` for clean disconnect
- `lib/useAgentStream.ts` — Custom hook: reads SSE chunks, `startTransition` throttles React re-renders
- `components/AgentTimeline.tsx` — Real-time SSE-driven agent status cards (running / done / error states)
- `components/ItineraryView.tsx` — Day-by-day accordion schedule with time slots
- `components/MapView.tsx` — **Mapbox GL JS** interactive map with destination markers + route polyline
- `components/DisruptionPanel.tsx` — weather, closure, and budget-cut controls
- `components/BudgetChart.tsx` — Cost breakdown bar chart (recharts)

---

## Repository Structure

```
backend/                 FastAPI + LangGraph
frontend/                Next.js 14 App Router
data/kerala_seed.json    Demo fallback dataset
PLAN.md
README.md
```

---

## Architecture verdict (locked)

Keep **exactly these 6 agents**. Do not add Hotel, Weather, Supervisor, or LLM Council agents.

That set is the right grain for a multi-agent trip planner:
- one parser, one explorer, one money optimizer, one mover, one scheduler, one critic
- hotels stay a **Budget tool**, weather stays a **Destination/Critic tool**
- Critic is the coordinator (routes which agent re-runs). A 7th supervisor would only add tokens

Do **not** fan-out Mobility and Budget in parallel. LangGraph would run Architect twice on first pass and can deadlock on targeted replan. Sequential Destination → Mobility → Budget also lets Budget include transport cost.

## PEAS (Review 1)

- **Performance**: feasible itinerary; maximize `Score = w_p P + w_q Q + w_r R + w_b B - w_t T - w_c C`; stay under budget and daily travel cap
- **Environment**: partially observable, dynamic, sequential, multi-agent; live APIs plus simulated disruptions
- **Actuators**: write TripState, emit itinerary, trigger targeted replan, ask user for missing fields
- **Sensors**: NL request, Mapbox, Foursquare, Gemini travel market, OpenWeatherMap, user HITL replies, disruption events

## Agent conflict protocol

When specialists disagree, they write an `AgentConflict` into shared state instead of overwriting each other:

```text
Destination: Munnar score 0.91
Budget: Munnar overshoots by 4000
Mobility: Munnar adds 3h travel
```

Critic picks the cheapest-damage repair (drop activity, cheaper hotel, swap city) and re-invokes only that agent.

## Offline fallback

`travel_market.py` asks Gemini, then Groq, for local hotels and airfares. If neither model is configured, it uses `data/kerala_seed.json` only when that city is in the file.

---

## Rubric Coverage

- **PEAS**: Performance (optimization score), Environment (real APIs, dynamic disruptions), Actuators (itinerary + replan), Sensors (weather, places, transport)
- **Agent Analysis**: 6 agents, partially observable + dynamic + multi-agent environment
- **Algorithmic Modeling**: A* (routing), OR-Tools CSP (scheduling), multi-objective weighted scoring
- **Tool Selection**: LangGraph (`MemorySaver`, conditional routing), LangChain `@tool`, FastAPI SSE, Mapbox, Gemini travel market, OpenWeatherMap, OR-Tools VRPTW, NetworkX, Foursquare Places
- **Multi-Agent Interaction**: Shared `TripState`, sequential specialists, Critic conflict resolution and targeted replan
- **Demo**: Part A — normal 5-day Kerala planning; Part B — live disruption simulation with targeted replan
