---
name: Odyssey Travel System
overview: Design and implement Odyssey — a 6-agent LangGraph-powered adaptive travel planning system with real API integrations (Mapbox, Open-Meteo) plus LLM-generated hotel/flight market data, weighted A* routing, OR-Tools VRPTW scheduling, and a Next.js frontend with live agent-progress streaming via SSE.
todos:
  - id: repo-init
    content: Initialize git repo, push to GitHub (akshayks13/Odyssey), set up monorepo structure
    status: completed
  - id: state-models
    content: Define TripState TypedDict, all Pydantic schemas (TripSpec, Destination, Activity, Route, Itinerary, ValidationReport, Disruption)
    status: completed
  - id: tools-layer
    content: "Implement all tools: mapbox_api.py (Geocoding/Directions), foursquare_api.py (Places/hours), travel_market.py (LLM hotels and flight/train quotes), weather.py (Open-Meteo), cost_calculator.py, preference_scorer.py, schedule_validator.py, budget_validator.py"
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
    content: "Next.js 14 App Router (nodejs runtime): landing page, live planning page with AgentTimeline (SSE via TransformStream proxy), ItineraryView (day accordion), MapView (MapLibre GL JS), DisruptionPanel, BudgetChart (recharts), useAgentStream hook"
    status: completed
  - id: tests
    content: Unit tests for A*, CSP, preference scorer, budget validator; agent-level tests with mocked tools; 3 full scenario integration tests (normal, weather disruption, budget overrun)
    status: completed
  - id: local-setup
    content: .env.example with API keys; README with local run instructions (uvicorn + npm run dev); a saved sample trip for demos when no model is available
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
- **LLM**: Gemini (`GEMINI_MODEL`, one or several keys that rotate) first; Groq (`GROQ_MODEL`) if every key is rate-limited or fails. Both run tool loops (Gemini through Google's `google-genai` SDK, because the LangChain client drops the thought signatures Gemini 3 needs)
- **LangGraph checkpointing**: `MemorySaver` for local/demo; optional `AsyncPostgresSaver` later if persistence is needed
- **Observability**: LangSmith (free tier) — trace every agent step, visualize tool calls, inspect reasoning chains; set via `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`
- **Algorithms**: weighted A* for city order; OR-Tools routing solver (VRPTW, guided local search) for the sights within each day; weighted-sum scorer (no scipy)
- **External APIs**:
  - **Mapbox**: geocoding and road directions (first choice; its free account can be paused, so each has one fallback)
  - **OSRM** (OpenStreetMap roads, free, no key): the drive-time fallback when Mapbox has no answer; its free-flow times are multiplied by 1.4
  - **OpenStreetMap tiles + MapLibre**: the interactive map in the browser (no key)
  - **Foursquare Places**: integrated in `foursquare_api.py` as a ready swap-in for `search_attractions` — held back by design, since the LLM path already covers entry fees and opening hours in one call across any country, which Foursquare's POI categories alone don't give
  - **LLM market data**: the primary source for sights (with entry fees and opening hours), hotels, flight/train quotes, and food prices — chosen over metered flight/hotel APIs (Amadeus, Duffel) after evaluating them, since it works for any region with no per-call cost or quota to manage during a demo
  - **Open-Meteo** (free, no key): a real forecast when the trip is within 16 days, otherwise the same dates last year as a seasonal guide, per day with rain chance
  - **Nominatim** (OpenStreetMap): free fallback geocoder when Mapbox has no answer
- **Database**: SQLite for saved itineraries and the graph state behind them (so a plan can be edited after a restart)
- **Cache**: in-process, per-tool, no TTL — one module-level dict each for geocode, directions, destinations, attractions, hotels, transport quotes and food cost; cleared between tests, not between server restarts
- **No model**: planning stops with a clear message; a saved sample trip (`backend/sample_trip.json`) opens without one
- **Frontend**: Next.js 14 App Router (nodejs runtime) + Tailwind (no component library) + MapLibre GL JS (OpenStreetMap tiles)
- **Streaming**: Server-Sent Events — FastAPI `StreamingResponse` + LangGraph `astream(version="v2", stream_mode=[...])` → Next.js `TransformStream` route handler
- **Deployment**: local run only

---

## The 6 Agents

Every specialist is a LangGraph node. The model **decides**; tools and algorithms **compute**. Without a model nothing is guessed: planning stops with a clear message.

Graph order (locked): Analyst → Destination → Mobility → Budget → Architect → Critic.

---

### Agent 1 — Trip Analyst (`backend/agents/trip_analyst.py`)

- **Role**: Parser. Turn the user's sentence into a `TripSpec`. Later agents plan; this one only extracts.
- **Input**: `raw_input` (natural language)
- **Output**: `trip_spec` (`destination_region`, `origin_city`, days, travellers, budget, preference weights, pace, `needs_clarification`)
- **Decides (LLM)**: Which place they asked to visit; home/arrival city if they said "from X to Y"; start date (it is told today's date); which of duration / travellers / budget it had to assume. If no destination was given it asks a question and the graph ends
- **Computes**: Validates the reply against `TripSpec` and keeps odd values in range. There is no fallback parser: without a model it stops with a clear error, because a guessed destination is worse than none
- **Must not**: Pick sights, hotels, a visit order, or a day plan. `origin_city` is arrival, not a sightseeing stop. Do not default the region to Kerala unless they asked for Kerala / a Kerala town or nothing was named.
- **Tools bound**: `geocode_location`, `validate_trip_schema`
- **Instruction**: SYSTEM_PROMPT in the file — "You are Odyssey's Trip Analyst — the parser, not the planner."

---

### Agent 2 — Destination Discovery (`backend/agents/destination_agent.py`)

- **Role**: Explorer. Choose which cities to keep as overnight/sightseeing stops and load attractions.
- **Input**: `trip_spec` (+ disruptions on replan)
- **Output**: `candidate_destinations`, `selected_destinations`, `candidate_activities`
- **Decides (LLM)**: Short list of city names from the ranking (never a disrupted city; never the origin/home city)
- **Computes**: Weighted cosine preference score; weather for the trip dates (forecast, or last year's dates); `max_destinations = duration_days // 2` (raised if the user named more cities). Honours cities the user pinned or excluded, and sees the Critic's reason when sent back
- **Must not**: Invent a schedule, pick hotels, or choose travel order (Mobility does A*)
- **Tools the model can call**: `search_attractions`, `get_directions`. The code itself uses:
  - `search_destinations` — the LLM lists places in the region (any country); nothing is invented if it cannot answer
  - `get_directions` — drive times, so the LLM can keep stops close together
  - `search_attractions` — LLM visitor sights, with entry fees and opening hours in the same call. Chosen over Foursquare's nearby search (`foursquare_api.py`, kept ready as a swap-in) because that returns every kind of venue — schools, shops, clinics — not just things worth visiting
  - weather (`trip_weather`, Open-Meteo) for each candidate, shown to the model so it can avoid a soaked stop
  - `score_preference_match` — cosine similarity
- **Instruction**: "You are Odyssey's Destination Discovery agent — the explorer, not the scheduler."

---

### Agent 3 — Mobility & Routing (`backend/agents/mobility_agent.py`)

- **Role**: Mover. Order the cities Destination already chose; label each hop road / rail / air.
- **Input**: `selected_destinations`, `trip_spec` (including optional `origin_city`)
- **Output**: `route` (ordered cities, legs, totals, A* stats)
- **Decides (LLM)**: The mode (road / rail / air) for every hop, including getting there and home. It knows which places have airports and when a flight is worth it, and asks for a quote first. Honours "by train / flight / road"
- **Computes**: NetworkX `DiGraph` of **road** times (Mapbox, else OSRM); weighted A* visit order (best of every start city). Hops: arrival `origin_city → first stop`, between stops, and the return home. Group prices: air and rail per seat × travellers. A road hop is priced for the whole group by the vehicle the model chooses for the trip (own car ₹8/km with tolls, hired taxi ₹12/km, tempo traveller ₹25/km per 12 seats, bus ₹2.5/km per person; cars counted per 4 people). It sees each vehicle's cost for this group next to the fares. It is told the trip budget, and prefers train or road when flights would eat it. **Guard**: a drive longer than the daily travel limit is replaced by the fastest flight or train that exists (Delhi to Chennai is never a drive) unless the traveller asked for road
- **Must not**: Add new cities, invent visit order, or pick hotels
- **Tools bound**: `search_public_transport` (LLM flight/train quote; a town without an airport is quoted with the ride to the nearest one), `check_transport_disruptions`. The code uses `get_directions`, `estimate_road_cost` and `calculate_route_cost`
- **Algorithms (not LLM tools)**: `build_travel_graph`, `astar_route_search` (`algorithms/astar.py`)
- **Instruction**: "You are Odyssey's Mobility agent — the mover, not the city picker."

---

### Agent 4 — Budget Optimization (`backend/agents/budget_agent.py`)

- **Role**: Money optimizer. Runs **after** Mobility so transport cost is already in the total.
- **Input**: `trip_spec`, `selected_destinations`, `route`, activities
- **Output**: `budget_breakdown`, `accommodation_options`, `excluded_activity_ids`, `stay_plan` (days and nights per city, shared with the Architect so the bill matches the schedule)
- **Decides (LLM)**: Hotel tier (budget/mid/premium), which picks the hotel (cheapest / middle / best rated); which activities to drop if over ceiling
- **Computes**: Line items (hotels per room × nights, food priced for the region, activities for the sights that fit, transport = the journeys plus a flat ₹500 a day per group of four for autos and cabs between the sights); `validate_budget`; cheapest-damage cuts keep ≥1 activity per city
- **Must not**: Reorder cities, invent rupee totals, or build the hour schedule. Hotel/flight quotes are market estimates, not bookable inventory.
- **Tools bound**: `search_hotels`, `estimate_food_costs`. The code uses `validate_budget` and `generate_tradeoff_options`
- **In-node (not bound)**: `calculate_activity_costs`, `estimate_local_transport`
- **Instruction**: "You are Odyssey's Budget agent — the money optimizer, not the scheduler."

---

### Agent 5 — Itinerary Architect (`backend/agents/itinerary_architect.py`)

- **Role**: Scheduler. Pack activities, meals, and overnight stays into days on the A* city order.
- **Input**: ordered cities, activities, route, budget (selected hotels)
- **Output**: `draft_itinerary`, `optimization_score`
- **Decides**: nothing by model. Pace (relaxed 2 sights 09–18 / moderate 3, 08–21 / packed 4, 07–22) comes from the Analyst's reading of the request
- **Computes**: Weather for each day (a rainy day schedules indoor sights first, outdoor ones wait for a dry day); the stay plan; time lost to arriving, long transfers and the trip home; opening hours; Mapbox travel matrix; OR-Tools VRPTW with meals; dated days; overnight hotel every night except the last; multi-objective score; replaces Budget's activity estimate with what was scheduled
- **Must not**: Reorder cities, pick hotels, or invent travel times
- **Tools bound**: none, no model call. The code uses the travel matrix and the solver
- **Algorithms (not LLM tools)**: `solve_day_schedule` (OR-Tools VRPTW), `compute_score`
- **In-node**: `check_schedule_conflicts` after the solve
- **Instruction**: "You are Odyssey's Itinerary Architect — the scheduler, not the router."

---

### Agent 6 — Critic & Replanner (`backend/agents/critic_replanner.py`)

- **Role**: Coordinator. Validate the draft; if invalid, re-invoke **one** specialist (max 3 loops). Never restarts Trip Analyst.
- **Input**: draft itinerary, budget, disruptions, `iteration_count`
- **Output**: `validation_report`, `replan_directives`, `final_itinerary` (when valid or out of loops)
- **Decides (LLM)**: Which of destination / mobility / budget / architect to call for the cheapest-damage repair
- **Computes**: Budget / schedule / closure / transport checks, plus "a day with nothing left to see" and "a transfer over the daily limit" (both go back to Destination with the reason, while a plan is first built, not during an edit). A long road trip to get there or home, a trip that is all travel with no sight, and a budget mostly spent on fares go back to Mobility. A hop with no route at all goes back to Destination, as does a city with no hotel even after Budget looked again (Destination must choose a nearby base and not the same city). An issue already retried is not sent again. The agent it goes back to sees the reason. Last allowed pass returns a best-effort plan.
- **Must not**: Rewrite days itself or fan-out every agent
- **Tools bound**: `check_weather_disruptions`, `check_transport_disruptions` (to verify a reported event). The code uses `validate_budget` and `check_schedule_conflicts`
- **Routing**: LangGraph conditional edges (`orchestration/routing.py`). The JSON reply **is** the replan directive — there is no separate `generate_replan_directive` tool.
- **Instruction**: "You are Odyssey's Critic & Replanner — the coordinator, not a seventh planner."

### Edit Router (`backend/agents/edit_router.py`)

- **Role**: Front door for changes to an existing plan. The Critic only reacts to problems it can detect, so it can't read "make day 2 lighter".
- **Input**: `edit_request` + the current plan. **Output**: updated `trip_spec`, standing `edit_locks`, `edit_directive`, or an `assistant_reply` for a question.
- **Decides (LLM)**: what changed, question vs change, and `route_to` (which agent to re-run).
- **Computes**: checks names against the plan; applies the change; the entry point may never be later than the earliest agent whose inputs changed.
- **Tools bound**: `get_plan_day`, `find_in_plan`, `list_alternative_cities`, `geocode_location`
- **Needs a model**: without one it leaves the plan alone and says so.

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

    # Stay plan (Budget writes it, the Architect schedules from it)
    stay_plan: List[StayBlock]

    # Prompt-based editing
    edit_request: Optional[str]
    edit_directive: Optional[EditDirective]
    edit_locks: EditLocks            # what the user has asked for; every agent reads it
    assistant_reply: Optional[str]   # answers and clarifying questions

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
  - **A* (meta-level)**: Finds the optimal order to visit N candidate destinations. Graph nodes = cities, edges = road travel time from Mapbox Directions (OSRM if Mapbox is unavailable). State = `(current_city, frozenset(visited), elapsed_days)`. Heuristic = min remaining travel between unvisited cities.
  - **Mapbox Directions API (road-level)**: Once order is decided, gives actual road route, duration, and polyline for the map.
- Weight escalation: `f(n) = g(n) + (1 + ε)·h(n)` — anytime behavior, always returns a plan
- Used by: Mobility Agent

### VRPTW Scheduling (`algorithms/csp_solver.py`)
- Library: `ortools.constraint_solver.pywrapcp.RoutingModel` (Vehicle Routing with Time Windows)
- Nodes = activities + hotels; edges = travel time between the day's stops (straight-line at city speed: the stops are a few km apart)
- Time windows per node = attraction opening hours (from the LLM's `search_attractions`, not Foursquare — see the External APIs note above)
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
# orchestration/graph.py — 7 nodes: the 6 below plus edit_router (added for prompt edits).
# START branches on whether this is a new plan or an edit (route_entry):
#   "plan"   -> trip_analyst
#   "revise" -> edit_router  (re-enters the pipeline at the shallowest agent the edit touches)
graph = StateGraph(TripState)
graph.add_node("trip_analyst", trip_analyst_node)
graph.add_node("destination_agent", destination_agent_node)
graph.add_node("mobility_agent", mobility_agent_node)
graph.add_node("budget_agent", budget_agent_node)
graph.add_node("itinerary_architect", itinerary_architect_node)
graph.add_node("critic_replanner", critic_replanner_node)
graph.add_node("edit_router", edit_router_node)

graph.add_conditional_edges(START, route_entry, {"plan": "trip_analyst", "revise": "edit_router"})
graph.add_conditional_edges("trip_analyst", route_after_analyst, {"go": "destination_agent", "ask": END})
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

- `POST /api/plan` → stashes the request in memory and returns `thread_id`; nothing runs until the client opens the stream below
- `GET /api/plan/{thread_id}/stream` → SSE stream; uses `graph.astream(version="v2", stream_mode=["messages","updates","custom"], subgraphs=True)`. Also the reconnect path: with nothing pending, it resumes the in-memory checkpoint, or — after a restart — rehydrates from the saved state, same as `/api/disrupt` below
- `POST /api/revise` → a free-text edit; the stream then starts at the Edit Router (from the stored state, so it works after a restart)
- `POST /api/disrupt` → injects `Disruption` into live state via `graph.aupdate_state()` → triggers Critic replan
- `GET /api/itinerary/{thread_id}` → fetch finalized itinerary from checkpoint
- `GET /api/health` → readiness check

SSE event types (`api/sse.py`):
```json
{"type": "init",          "thread_id": "..."}
{"type": "step_start",    "agent": "destination_agent", "message": "destination_agent started..."}
{"type": "tool_result",   "agent": "destination_agent", "tool": "search_attractions", "data": {...args}}
{"type": "step_complete", "agent": "itinerary_architect", "message": "...", "meta": {...}, "data": {...}}
{"type": "custom", ...}    // reserved for a node-level progress event; no node emits one yet
{"type": "done", "itinerary": {...}, "budget": {...}, "route": {...}, "score": 0.87, ...}
{"type": "error",         "message": "...", "thread_id": "..."}
```

Key FastAPI pattern: graph compiled **once at startup** via `lifespan`; `thread_id = uuid4()` (no user/session concept — a client may also supply its own on `POST /api/plan`); `X-Accel-Buffering: no` header for Nginx.

---

## Frontend (Next.js)

- `app/page.tsx` — Hero + natural language input form
- `app/plan/[threadId]/page.tsx` — Live planning view with agent progress + map
- `app/api/stream/route.ts` — Next.js Route Handler (`runtime = 'nodejs'`); proxies FastAPI SSE via `TransformStream`; forwards `req.signal` for clean disconnect
- `lib/useAgentStream.ts` — Custom hook: reads SSE chunks, `startTransition` throttles React re-renders
- `components/AgentTimeline.tsx` — Real-time SSE-driven agent status cards (running / done / error / kept states), live tool-call line, tool/algorithm/engine chips
- `components/ItineraryView.tsx` — Day-by-day accordion schedule with time slots
- `components/MapView.tsx` — **MapLibre GL JS** interactive map (OpenStreetMap tiles) with destination markers + route polyline
- `components/DisruptionPanel.tsx` — weather, closure, transport, and budget-cut controls
- `components/BudgetChart.tsx` — Cost breakdown bar chart (recharts)
- `components/EditBox.tsx` — chat-style transcript + free-text change request box (prompt edits)
- `components/Header.tsx` / `Footer.tsx` / `Wordmark.tsx` — site chrome

---

## Repository Structure

```
backend/                 FastAPI + LangGraph
frontend/                Next.js 14 App Router
backend/sample_trip.json A saved plan for demos when no model is available
PLAN.md
README.md
```

---

## Architecture verdict (locked)

Keep **these 6 planning agents**, plus the **Edit Router** for changes after a plan exists. Do not add Hotel, Weather, Supervisor, or LLM Council agents.

That set is the right grain for a multi-agent trip planner:
- one parser, one explorer, one money optimizer, one mover, one scheduler, one critic
- hotels stay a **Budget tool**, weather stays a **Destination/Critic tool**
- Critic is the coordinator (routes which agent re-runs). A 7th supervisor would only add tokens

Do **not** fan-out Mobility and Budget in parallel. LangGraph would run Architect twice on first pass and can deadlock on targeted replan. Sequential Destination → Mobility → Budget also lets Budget include transport cost.

## PEAS (Review 1)

- **Performance**: feasible itinerary; maximize `Score = w_p P + w_q Q + w_r R + w_b B - w_t T - w_c C`; stay under budget and daily travel cap
- **Environment**: partially observable, dynamic, sequential, multi-agent; live APIs plus simulated disruptions
- **Actuators**: write TripState, emit itinerary, trigger targeted replan, ask user for missing fields
- **Sensors**: NL request, Mapbox / OSRM, Open-Meteo, LLM market data, user HITL replies, disruption events

## Agent conflict protocol

When specialists disagree, they write an `AgentConflict` into shared state instead of overwriting each other:

```text
Destination: Munnar score 0.91
Budget: Munnar overshoots by 4000
Mobility: Munnar adds 3h travel
```

Critic picks the cheapest-damage repair (drop activity, cheaper hotel, swap city) and re-invokes only that agent.

## One fallback each

| Need | Source | Fallback |
|---|---|---|
| Model | Gemini (each key in turn) | Groq; if all are rate-limited, planning stops with a clear message |
| Places, sights, hotels, flights and trains | the model | none |
| Geocoding | Mapbox | Nominatim, then "not found" |
| Drive times | Mapbox | OSRM (times x1.4), then "unavailable" |
| Food prices | the model | one flat default per tier |
| Weather | Open-Meteo | none ("Weather unavailable") |

With no model the demo path is the saved sample trip (`POST /api/sample`).

---

## Rubric Coverage

- **PEAS**: Performance (optimization score), Environment (real APIs, dynamic disruptions), Actuators (itinerary + replan), Sensors (weather, places, transport)
- **Agent Analysis**: 6 agents, partially observable + dynamic + multi-agent environment
- **Algorithmic Modeling**: A* (routing), OR-Tools CSP (scheduling), multi-objective weighted scoring
- **Tool Selection**: LangGraph (`MemorySaver`, conditional routing), LangChain `@tool`, FastAPI SSE, Mapbox, OSRM, OpenStreetMap/MapLibre, Open-Meteo, LLM market data, OR-Tools VRPTW, NetworkX, Foursquare Places
- **Multi-Agent Interaction**: Shared `TripState`, sequential specialists, Critic conflict resolution and targeted replan
- **Demo**: Part A — normal 5-day Kerala planning; Part B — live disruption simulation with targeted replan
