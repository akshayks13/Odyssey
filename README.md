# Odyssey — Multi-Agent Adaptive Travel Planning System

Odyssey turns a natural-language trip request into a constraint-aware itinerary using **six specialized LangGraph agents** that share a single `TripState`. When weather, closures, transport, or budget change, the Critic routes a **targeted replan** to only the affected agent instead of regenerating the whole trip.

The locked design is in [`PLAN.md`](PLAN.md). This README is how to run it.

```
User
  → 1. Trip Analyst
  → 2. Destination Discovery
  → 3. Mobility & Routing  (weighted A*)
  → 4. Budget Optimization
  → 5. Itinerary Architect (OR-Tools VRPTW)
  → 6. Critic & Replanner
         ├─ valid → final itinerary
         ├─ destination issue → Destination Agent
         ├─ route issue → Mobility Agent
         ├─ budget issue → Budget Agent
         └─ schedule issue → Itinerary Architect
```

## What each agent does

Every specialist is a LangGraph node. When `GEMINI_API_KEY` or `GROQ_API_KEY` is set, that node runs the same ReAct loop (`llm.bind_tools` → tool calls → JSON decision, up to 6 rounds). The model **chooses**; deterministic tools and algorithms **compute**. With no LLM key (or `ODYSSEY_DISABLE_LLM=1` in tests), the same tools still run on a heuristic path so tests stay offline.

Full role / decides / computes / tools cards (kept in sync with the code) are in [`PLAN.md`](PLAN.md#the-6-agents). Each agent file also starts with that contract.

| # | Agent | Role | Decides (LLM) | Must not | Tools bound | Computes |
|---|---|---|---|---|---|---|
| 1 | **Trip Analyst** | Parser | Region, origin (arrival only), days, budget, prefs | Pick sights, hotels, or a day plan | `geocode_location`, `validate_trip_schema` | Regex heuristic if no LLM |
| 2 | **Destination Discovery** | Explorer | Which cities to keep (not origin, not disrupted) | Order the route or book hotels | `search_destinations`, `search_attractions`, `get_weather_forecast`, `score_preference_match`, `get_place_photos` | Cosine score; attractions per city |
| 3 | **Mobility & Routing** | Mover | Start city + hop mode label | Add cities or invent order | `quote_transport`, `search_flights`, `check_transport_availability`, `calculate_route_cost` | Mapbox road graph + weighted A* |
| 4 | **Budget Optimization** | Money | Hotel tier, cheapest vs rated, activity cuts | Reorder cities or invent rupee totals | `search_hotels`, `search_hotel_offers`, `estimate_food_costs`, `validate_budget`, `generate_tradeoff_options` | Line items vs ceiling |
| 5 | **Itinerary Architect** | Scheduler | Pace (relaxed / moderate / packed) | Reorder cities or pick hotels | `get_opening_hours`, `travel_time_matrix`, `validate_time_windows` | OR-Tools VRPTW + score |
| 6 | **Critic & Replanner** | Coordinator | Which one specialist to re-invoke | Restart Analyst or rewrite days | `validate_budget`, `check_schedule_conflicts`, `check_transport_disruptions`, `check_attraction_availability`, `check_weather_disruptions` | Severity sort + graph edges |

**24 tool bindings** across the six agents (`validate_budget` is shared by Budget and Critic). Graph order is sequential: Analyst → Destination → Mobility → Budget → Architect → Critic.

## LLM reasoning vs tools vs seed data

This is **not** a mock-only pipeline when keys are present:

| Layer | What it does |
|---|---|
| **LLM (Gemini Flash-Lite, Groq fallback)** | ReAct tool-calling. Decides *what* to search, *which* cities/hotels to keep, *start city*, *pace*, *which agent to re-invoke*. |
| **Deterministic tools** | A*, OR-Tools VRPTW, cost/budget arithmetic, preference cosine, validators. The model cannot invent rupee totals. |
| **Live APIs** | Mapbox, Foursquare, OpenWeatherMap, Gemini (hotels/flights + agent tool-calling) — used first when keys exist. |
| **`kerala_seed.json`** | Offline rows for cities in that file when a live lookup has nothing for them. |

Without `GEMINI_API_KEY` / `GROQ_API_KEY`, agents fall back to heuristics + seed so tests and offline demos work. With a key, each specialist runs `bind_tools` and the timeline messages say `via gemini:… tool-calling`. `/api/health` reports `"llm": null` when no key is loaded.

## LangSmith

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT=odyssey`). Every LangGraph node, ReAct loop, and tool call is traced.

Open the project: [odyssey on LangSmith](https://smith.langchain.com/o/d8cb6e0e-760d-44d4-bfa5-12c595e8d471/projects/p/bcac7ca7-bdd6-45bb-9478-471f959f49de)

Latest live pipeline run (public): [odyssey:live-full-pipeline](https://smith.langchain.com/public/998433d4-9a8f-4383-83a9-d0a3bfcc5548/r)

## Prerequisites

- Python 3.11+ (tested on 3.13)
- Node.js 18+ (tested on 22)
- Optional API keys (Gemini / Groq, Mapbox, Foursquare, OpenWeatherMap, LangSmith)

## Local run

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # fill keys if you have them; leave blank to use seed data
uvicorn api.main:app --reload --port 8000
```

Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_MAPBOX_TOKEN for the interactive map
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 3. Tests

```bash
cd backend
source .venv/bin/activate
pytest -q
```

Covers A*, CSP scheduling, preference/budget scorers, each agent, and three full scenarios (normal plan, closure replan, budget-cut replan).

## Demo script (reviews)

**Part A — normal planning**

Enter:

> 5-day Kerala trip for 4 people, ₹40,000, nature + adventure, relaxed pace.

Watch the six agents stream live, then inspect the day accordion, budget chart, and route.

**Part B — live disruption**

Click **Close Munnar** (or Weather / Transport / Cut Budget). The Critic flags the issue and re-invokes only the responsible agent. Example: a Munnar closure swaps Day 2 to Vagamon/Wayanad without restarting Trip Analyst.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/plan` | Start a planning session, returns `{thread_id}` |
| `GET` | `/api/plan/{thread_id}/stream` | SSE stream (`astream` messages/updates/custom) |
| `POST` | `/api/disrupt` | Inject a disruption; then reconnect to the stream |
| `GET` | `/api/itinerary/{thread_id}` | Fetch current itinerary (checkpoint or SQLite) |
| `GET` | `/api/health` | Readiness |

The Next.js app proxies the SSE stream at `GET /api/stream?threadId=` via a `TransformStream`.

SSE event types: `init`, `step_start`, `tool_result`, `step_complete`, `done`, `error`.

## Optimization model

\[
Score = w_p P + w_q Q + w_r R + w_b B - w_t T - w_c C
\]

Weights are personalized by traveller archetype (`adventure` / `relaxed` / `budget` / `balanced`), inferred from the Trip Analyst preference profile.

## Repository layout

```
backend/                 FastAPI + LangGraph agents, tools, algorithms
frontend/                Next.js 14 App Router + Tailwind
data/kerala_seed.json    Offline city/activity/hotel rows for cities in that file
```

## Environment variables

See `backend/.env.example` and `frontend/.env.local.example`. Keys are optional; bundled seed rows cover cities present in that file.
