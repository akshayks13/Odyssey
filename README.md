# Odyssey — Multi-Agent Adaptive Travel Planning System

Odyssey turns a natural-language trip request into a constraint-aware itinerary using **six specialized LangGraph agents** that share a single `TripState`. When weather, closures, transport, or budget change, the Critic routes a **targeted replan** to only the affected agent instead of regenerating the whole trip. After a plan exists you can also change it by typing ("make day 2 lighter", "add Alleppey"): an **Edit Router** reads the request and re-runs only the agents it touches.

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

Every specialist is a LangGraph node. When `GEMINI_API_KEY` or `GROQ_API_KEY` is set, that node runs the same ReAct loop (`llm.bind_tools` → tool calls → JSON decision, a few rounds, then the final answer). The model **chooses**; deterministic tools and algorithms **compute**. Without a model there is no guessed plan: it stops with a clear message (see "When no model is available").

Full role / decides / computes / tools cards (kept in sync with the code) are in [`PLAN.md`](PLAN.md#the-6-agents). Each agent file also starts with that contract.

| # | Agent | Role | Decides (LLM) | Must not | Tools bound | Computes |
|---|---|---|---|---|---|---|
| 1 | **Trip Analyst** | Parser | Region, origin (arrival only), days, budget, prefs | Pick sights, hotels, or a day plan | `geocode_location`, `validate_trip_schema` | Validates the reply; stops with a clear error if no model is available |
| 2 | **Destination Discovery** | Explorer | Which cities to keep (not origin, not disrupted) | Order the route or book hotels | `search_attractions`, `get_directions` | Places (LLM), weather for the trip dates, cosine score |
| 3 | **Mobility & Routing** | Mover | Road / rail / air for every hop, including there and home (never a drive over the daily limit unless asked), and the road vehicle: own car, taxi, tempo traveller or bus | Add cities or invent order | `search_public_transport`, `check_transport_disruptions` | Road-time graph (Mapbox, else OSRM) + weighted A* |
| 4 | **Budget Optimization** | Money | Hotel tier, activity cuts | Reorder cities or invent rupee totals | `search_hotels`, `estimate_food_costs` | Line items vs ceiling |
| 5 | **Itinerary Architect** | Scheduler | Nothing by model: pace comes from the Analyst | Reorder cities or pick hotels | none | OR-Tools VRPTW, weather per day, score |
| 6 | **Critic & Replanner** | Coordinator | Which one specialist to re-invoke | Restart Analyst or rewrite days | `check_weather_disruptions`, `check_transport_disruptions` | Severity sort + graph edges |
| 7 | **Edit Router** | Front door for changes | What changed, question vs change, which agent to re-run | Rewrite the plan itself | `get_plan_day`, `find_in_plan`, `list_alternative_cities`, `geocode_location` | Name checks; never re-enters after the earliest agent whose inputs changed |

Graph order for a new plan is sequential: Analyst → Destination → Mobility → Budget → Architect → Critic. The Critic can't handle "make day 2 lighter" because nothing is wrong to detect, which is why edits have their own agent. Everything it learns is stored as standing instructions (`EditLocks`) that every agent reads, so later replans keep the edit.

## LLM reasoning vs tools

| Layer | What it does |
|---|---|
| **LLM (Gemini, then Groq)** | ReAct tool-calling. Decides *which* cities/hotels to keep, *how to travel each hop*, *which agent to re-invoke*, and how to read an edit. Also supplies places for any region, sights with entry fees, hotels, flight/train quotes, food prices. |
| **Deterministic tools** | A*, OR-Tools VRPTW, cost/budget arithmetic, preference cosine, validators. The model cannot invent rupee totals. |
| **Live APIs** | Mapbox (geocoding and drive times, first choice), Open-Meteo (weather: the forecast up to 16 days ahead, otherwise the same dates last year; free, no key), Nominatim (the one geocoding fallback), OSRM (the one drive-time fallback: OpenStreetMap roads, free, no key; times are multiplied by 1.4 because it assumes free-flowing traffic and the leg is marked "estimated"). The map in the browser is MapLibre on OpenStreetMap tiles: no key. Times between sights within a day are straight-line at city speed. Foursquare is kept but not used as a source of sights. |

Gemini is tried first; on a rate limit the call goes to the next key, and Groq is the last resort. `GEMINI_API_KEY` can hold several comma-separated keys, and each key from a separate Google project has its own quota, so more keys mean more plans per day and less waiting. All can call tools. Reading your request and editing a plan need a model; if none is available (a rate limit) it says so instead of guessing. There is no seed data. With a key, each specialist runs `bind_tools` and the timeline messages say `via gemini:… tool-calling`. `/api/health` reports the model in use, or `"unavailable"`.

### When no model is available

The free tiers run out (Gemini per minute and per day for each key, Groq within a couple of plans). Planning then stops with "The language model is unavailable…" instead of a made-up plan, and an edit says the plan is unchanged. For a demo, **See a sample trip** (landing page, and on that error) opens `backend/sample_trip.json`, a saved plan (the Kerala test data with real drive times and weather), through `POST /api/sample`. It needs no model, and can be edited once one is available again.

## LangSmith

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT=odyssey`). Every LangGraph node, ReAct loop, and tool call is traced.

Open the project: [odyssey on LangSmith](https://smith.langchain.com/o/d8cb6e0e-760d-44d4-bfa5-12c595e8d471/projects/p/bcac7ca7-bdd6-45bb-9478-471f959f49de)

Latest live pipeline run (public): [odyssey:live-full-pipeline](https://smith.langchain.com/public/998433d4-9a8f-4383-83a9-d0a3bfcc5548/r)

## Prerequisites

- Python 3.11+ (tested on 3.13)
- Node.js 18+ (tested on 22)
- API keys: Groq and/or Gemini (at least one, required for planning; several Gemini keys can be comma-separated). Mapbox and LangSmith are optional. Weather and the map need no key.

## Local run

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # fill keys if you have them; at least one of GROQ_API_KEY / GEMINI_API_KEY is needed to plan
uvicorn api.main:app --reload --port 8000
```

Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # only the API address; the map needs no key
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 3. Tests

```bash
cd backend
source .venv/bin/activate
pytest -q
```

Runs offline in about 3 minutes (no keys, no network): a small Kerala data set stands in for the model and geocoder, so the real code paths run. Covers A*, the scheduler, the tools, the provider chain, each agent, the Edit Router, plan consistency and the full scenarios.

## Demo script (reviews)

**Part A — normal planning**

Enter:

> 5-day Kerala trip for 4 people, ₹40,000, nature + adventure, relaxed pace.

Watch the six agents stream live, then inspect the day accordion, budget chart, and route.

**Part B — live disruption**

Click **Close <city>** (or Weather / Transport / Cut Budget). The Critic flags the issue and re-invokes only the responsible agent: closing a city replans with another nearby one, without restarting the Trip Analyst. This also works on a plan reopened after a server restart.

**Part C — change it with a sentence**

Type "make day 2 lighter" or "add Alleppey" in the **Change anything** box. The Edit Router decides which agent to re-run, and the timeline shows the agents that were kept unchanged.

**If the model is rate-limited during a demo**

The app says so instead of guessing. Click **See a sample trip** (landing page, or the link in the error box) to open a saved plan that needs no model.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness (`{name, status, docs}`); interactive docs at `/docs` |
| `POST` | `/api/plan` | Start a planning session, returns `{thread_id}` |
| `GET` | `/api/plan/{thread_id}/stream` | SSE stream (`astream` messages/updates/custom). Also the reconnect path after a reload or a restart — resumes the in-memory checkpoint, or rehydrates from the saved state if that's empty |
| `POST` | `/api/sample` | Open the saved sample trip as a new thread (needs no model), returns `{thread_id}` |
| `POST` | `/api/revise` | Change the plan with a sentence (`{thread_id, message}`); then reconnect to the stream |
| `POST` | `/api/disrupt` | Inject a disruption; then reconnect to the stream |
| `GET` | `/api/itinerary/{thread_id}` | Fetch current itinerary (checkpoint or SQLite) |
| `GET` | `/api/health` | Readiness |

The Next.js app proxies the SSE stream at `GET /api/stream?threadId=` via a `TransformStream`.

SSE event types: `init`, `step_start`, `tool_result`, `step_complete`, `custom` (reserved — no node emits one yet), `done`, `error`.

## Optimization model

\[
Score = w_p P + w_q Q + w_r R + w_b B - w_t T - w_c C
\]

Weights are personalized by traveller archetype (`adventure` / `relaxed` / `budget` / `balanced`), inferred from the Trip Analyst preference profile.

## Repository layout

```
backend/                 FastAPI + LangGraph agents, tools, algorithms
frontend/                Next.js 14 App Router + Tailwind
backend/sample_trip.json A saved plan for the demo (no model needed)
```

## Environment variables

See `backend/.env.example` and `frontend/.env.local.example`. Keys are optional. `GEMINI_MODEL` is the primary model; `GROQ_MODEL` is the last-resort fallback.

## Review documents

- [`docs/review1_design_report.md`](docs/review1_design_report.md): PEAS, environment and agent analysis, algorithmic modelling and search strategy, Q&A
- [`docs/review2_implementation.md`](docs/review2_implementation.md): tools and setup, multi-agent execution, demo and testing scenarios, code structure
