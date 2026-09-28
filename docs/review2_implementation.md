# Odyssey — Review 2: Implementation

Branch `main` (LLM agents with live APIs). The rule-based, no-LLM version is on the branch `simple-deterministic`.

Rubric lines covered: **Tool/package selection and setup** (§1) · **Multi-agent execution and interaction** (§2) · **Demo quality and testing scenarios** (§3) · **Code structure and scalability** (§4). The design behind this is in [`review1_design_report.md`](review1_design_report.md).

## 1. Tool / package selection and setup

### What is used and why

| Tool | Version | Used for | Why this one |
|---|---|---|---|
| Python | 3.11+ (tested on 3.13) | Backend | Course language; the agents, algorithms and tools are plain Python |
| FastAPI + uvicorn | 0.115.6 / 0.32.1 | HTTP API and server-sent events | Async streaming of each agent's progress to the UI with little code |
| LangGraph | 0.2.60 | The agent graph: shared typed state, conditional edges for the Critic's routing, in-memory checkpoints | The Critic's "send the plan to one agent" is a conditional edge; a checkpoint lets a disruption or an edit resume the same plan |
| LangChain core, `google-genai`, `langchain-groq` | 0.3.28 / 2.24.0 / 0.2.1 | Talking to the models with tools bound | Gemini goes through Google's own SDK because the LangChain client drops the thought signatures Gemini 3 needs; Groq is the last resort |
| LangSmith | 0.2.11 | Tracing every node, tool call and model round | The evidence of what each agent did on a live run |
| Pydantic | 2.13.5 | Schemas for every piece of state and every model reply | A bad model reply is rejected, not trusted |
| OR-Tools | 9.12 | Routing with time windows for each day | Models opening hours, meals and optional visits directly |
| NetworkX | 3.4.2 | The road-time graph the A\* search runs on | Standard graph structure |
| SQLAlchemy + SQLite | 2.0.36 | Saved itineraries and the state behind them | A plan can be edited after a restart |
| httpx | 0.28.1 | Calls to the map, weather and geocoding services | Async HTTP |
| pytest + pytest-asyncio | 8.3.4 / 0.25.0 | Tests | One command runs everything offline |
| Next.js 14, Tailwind, MapLibre GL, Recharts, framer-motion | see `frontend/package.json` | The web UI | Streaming timeline, day-by-day view, map and budget chart |

### External services

| Service | Role | If it fails |
|---|---|---|
| Gemini (one or several keys) | The model, first choice | Rate limit → the next key → Groq |
| Groq | The model, last resort | If every model fails, planning stops with a clear message |
| Mapbox | Geocoding and road directions | OpenStreetMap geocoder (Nominatim) and OSRM (times × 1.4); otherwise "unavailable", never invented |
| Open-Meteo | Weather per trip day (forecast within 16 days, else the same dates last year) | The plan continues without weather |
| OpenStreetMap tiles | The map in the browser | The route is shown as a list |
| The model itself | Places, sights with fees and hours, hotels, flight and train quotes, food prices | No fallback: without a model nothing is invented |

### Considered and not used

| Option | Why not |
|---|---|
| Amadeus / Duffel (flight and hotel APIs) | Metered, with quotas to manage during a demo; the model path works for any region at no per-call cost |
| Foursquare Places for sights | Returns every kind of venue (schools, shops, clinics), not only things worth visiting; it is integrated but unused |
| Running Mobility and Budget in parallel | LangGraph would run the Architect twice on the first pass and can deadlock on a targeted replan; sequential also lets Budget include the fares |
| Extra agents (Hotel, Weather, Supervisor, LLM Council) | Hotels are a Budget tool and weather is a Destination and Critic tool; a supervisor would only add cost |

### Setup

Requires Python 3.11+ and Node 18+. At least one model key is required to plan: Gemini and/or Groq. The map and weather need no key.

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill GEMINI_API_KEY (several keys may be comma-separated) and/or GROQ_API_KEY
uvicorn api.main:app --reload --port 8000        # http://localhost:8000/docs

# frontend (second terminal)
cd frontend
npm install
cp .env.local.example .env.local                 # only the backend address
npm run dev                                       # http://localhost:3000
```

| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Primary model; several keys separated by commas rotate on a rate limit |
| `GROQ_API_KEY`, `GROQ_MODEL` | Last-resort model |
| `MAPBOX_API_KEY` | Optional; better geocoding and drive times |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional tracing |
| `MAX_REPLAN_ITERATIONS` | How many times the Critic may send a plan back (default 3) |

`GET /api/health` reports which model is in use, or `"unavailable"`.

## 2. Multi-agent execution and interaction

### How agents interact

- **One shared state.** The six specialists are LangGraph nodes over a single `TripState`. Each reads what earlier agents wrote and writes its own part: `trip_spec`, then `selected_destinations` and `candidate_activities`, then `route`, then `budget_breakdown` and `stay_plan`, then `draft_itinerary`, then `validation_report`.
- **A fixed order for a new plan.** Analyst → Destination → Mobility → Budget → Architect → Critic. Budget runs after Mobility so fares are already in the total.
- **The Critic coordinates.** After every plan it validates and, if something is wrong, names exactly one specialist to re-run. LangGraph conditional edges (`orchestration/routing.py`) carry that decision. Three loops at most, then the best plan so far is returned with its issues listed.
- **The agent it is sent back to is told why.** The Critic's reason is passed along (tested for Destination and Mobility).
- **Edits go through a front door.** The Edit Router reads a typed change, stores it as standing choices (`EditLocks`, which every agent reads so a later replan keeps the edit) and re-enters the graph at the earliest agent whose inputs changed. It may re-run more than needed, never less.

### Who fixes what (the Critic)

| Issue found | Sent to |
|---|---|
| Reported closure or storm at a city | Destination |
| A hop longer than the daily limit, or a day with nothing left to see (while a plan is first built) | Destination, with the reason |
| A hop with no route at all | Destination |
| A city with no hotel even after Budget looked again | Destination, told to pick a nearby base and not the same city |
| Over budget | Budget (Mobility if fares are most of the budget) |
| A night with no hotel | Budget |
| A long road trip to get there or home; a trip that is all travel; fares eat the budget | Mobility |
| Schedule overlap | Itinerary Architect |
| Reported strike | Mobility |

Rules around it: a reported event is handled before an unrelated problem; an issue already retried is not sent again; an edit does not send the plan back to pick new cities.

### Conflicts

When specialists disagree they write an `AgentConflict` into the state instead of overwriting each other, for example Destination's favourite city, Budget's overrun, and Mobility's extra travel time. The Critic picks the cheapest-damage repair (cheaper hotel, drop a sight, swap the city) and re-invokes only that agent.

### What the model does inside an agent

Each agent binds its own tools to the model, lets it call them for a few rounds, and validates a JSON decision. Examples: Mobility asks for a quote before choosing rail or air; Destination calls `search_attractions` on a shortlisted city; the Critic can verify a reported storm before choosing where to send the plan. Every tool call is visible in the UI timeline and in LangSmith.

### Evidence

- Live view: the timeline shows each agent, its message, the tools it called and the algorithms it ran; a run that only touched some agents shows the others as "unchanged".
- Traces: [LangSmith project](https://smith.langchain.com/o/d8cb6e0e-760d-44d4-bfa5-12c595e8d471/projects/p/bcac7ca7-bdd6-45bb-9478-471f959f49de) and a [public live run](https://smith.langchain.com/public/998433d4-9a8f-4383-83a9-d0a3bfcc5548/r).
- Tests of routing and re-entry are listed in §3.

### An honest note

Agents cooperate through the shared state and the graph. What each agent may not do ("must not reorder cities", "must not invent totals") is stated in its instructions and backed by code that recomputes every number; the runtime does not stop an agent from writing a field it does not own. The branch `simple-deterministic` enforces ownership and uses explicit messages.

## 3. Demo quality and testing scenarios

### Demo script

**Part A — normal planning.** Enter: *5-day Kerala trip for 4 people, ₹40,000, nature + adventure, relaxed pace.* Watch the six agents stream. Open a day, the budget chart and the route map. Point at the chips under each step: the tools it called and the algorithm it ran (A\* nodes, OR-Tools days solved).

**Part B — a live disruption.** Click *Close ⟨city⟩* (or Weather, Transport, Cut budget). The Critic names one agent; closing a city re-runs Destination onward without restarting the Analyst. This also works on a plan reopened after a server restart.

**Part C — change it with a sentence.** Type "make day 2 lighter" or "add Alleppey" in *Change anything*. The Edit Router picks the entry agent; the timeline shows the agents that were kept.

**If the model is rate-limited.** The app says so instead of guessing. *See a sample trip* opens a saved plan that needs no model.

### The test suite

Run offline, with no keys and no network:

```bash
cd backend && source .venv/bin/activate && pytest -q      # 189 tests, about 3 minutes
```

A scripted model and a small fake Kerala world (`tests/data/kerala_world.json`) stand in for the network, so the real code paths run.

| File | Test functions | What it checks |
|---|---|---|
| `test_algorithms.py` | 31 | A\* (shortest order, always returns a plan, routes around an unavailable leg, respects the daily cap); the day scheduler (feasible, no overlaps, opening hours, meals, the morning is used); stay plan; cosine, budget, score |
| `test_agents.py` | 41 | Each agent (Analyst, Destination, Mobility, Budget, Architect); the routing function; the API's plan and stream endpoints; saving; a corrupt saved state |
| `test_edit_flow.py` | 34 | Edits re-enter at the right agent; the Critic's routing rules; a restart; a reconnect |
| `test_tools.py` | 32 | Geocoding and directions with their fallbacks; weather; prices; the model provider chain (rate limits, failover, malformed replies) |
| `test_scenarios.py` | 18 | Whole-pipeline scenarios and plan-consistency checks |

Parametrised cases bring the total to 189.

### Scenarios

| # | Scenario | Expected | Test |
|---|---|---|---|
| 1 | Normal 5-day Kerala trip | Six agents, a valid plan within budget | `test_scenario_normal_kerala_trip` |
| 2 | A city closes | Critic sends the plan to Destination only; the other cities stay | `test_scenario_weather_closure_replans_destination` |
| 3 | The budget is cut | Critic sends the plan to Budget | `test_scenario_budget_cut_triggers_budget_agent` |
| 4 | The plan is checked as a whole | Consistent dates, hotels, costs; no sight free by accident | `test_the_plan_is_internally_consistent`, `test_no_sight_is_free_by_accident` |
| 5 | "Make day 2 lighter" | Only the Architect re-runs | `test_a_packed_day_reruns_only_the_scheduler` |
| 6 | "Add a city" / "cheaper hotels" | Re-enter at Destination / Budget | `test_adding_a_city_reenters_at_destination`, `test_hotel_edit_reenters_at_budget` |
| 7 | The router tries to skip an agent whose inputs changed | It cannot | `test_the_router_cannot_skip_an_agent_whose_inputs_changed` |
| 8 | A question | Answered; plan unchanged | `test_a_question_changes_nothing` |
| 9 | An issue was already retried | Not sent again | `test_an_issue_that_was_already_retried_is_not_sent_again` |
| 10 | A night with no hotel; a hop with no route | Budget; Destination | `test_a_night_without_a_hotel_goes_back_to_budget`, `test_a_hop_with_no_route_goes_back_to_destination_…` |
| 11 | The server restarts | A saved plan can still be edited | `test_editing_works_after_a_restart` |
| 12 | A tiny expansion budget | A\* still returns a plan | `test_astar_always_returns_a_plan_even_with_tiny_expansion_budget` |
| 13 | No model available | A clear message; nothing invented | `test_the_plan_stops_with_a_clear_message_when_no_places_come_back` |
| 14 | A model rate limit | Next key, then Groq | `test_a_rate_limited_gemini_key_hands_over_to_the_next_key`, `test_rate_limited_groq_falls_back_to_gemini_…` |
| 15 | Mapbox has no answer | OSRM, times stretched; else "unavailable" | `test_osrm_answers_when_mapbox_does_not_…` |

### What the evidence does and does not show

Shows: the pipeline runs end to end, each routing rule sends a problem to the right agent, algorithm properties hold, and failures of the model and the services degrade to a clear message. A traced live run is public (§2).

Does not show: a quantitative comparison against baselines (independent agents, greedy choices) or results over many seeds. Live results also vary with the model and with the forecast. That comparison is done on the branch `simple-deterministic`.

## 4. Code structure and scalability

```
backend/
  agents/          one module per agent (trip_analyst, destination_agent, mobility_agent, budget_agent,
                   itinerary_architect, critic_replanner, edit_router)
  algorithms/      astar.py (visiting order), csp_solver.py (OR-Tools day timetable), optimizer.py (score), planning.py
  tools/           the tools the agents call: mapbox_api, foursquare_api, travel_market, weather,
                   cost_calculator, budget_validator, schedule_validator, preference_scorer
  orchestration/   graph.py (the LangGraph graph), routing.py (conditional edges), state.py (TripState)
  models/          schemas.py: every Pydantic model
  api/             FastAPI routes and the event stream
  services/        SQLite persistence of plans
  llm.py           model providers, tool-calling loop, failover
  tests/
frontend/          Next.js 14 app: landing page, live plan page, timeline, itinerary, map, budget chart, edit box
PLAN.md  README.md  docs/
```

About 4,600 lines of Python across the agents, algorithms, tools, graph, API and model layer. One concern per module: algorithms are pure functions with no model or network in them and are tested directly; tools are the only code that touches a service.

**Scalability**

| To add | What to do |
|---|---|
| A region or country | Nothing in code: places, sights, hotels and prices come from the model and the map services |
| A tool | Write it in `tools/`, bind it to the agent that needs it |
| An agent | Write a node in `agents/`, add it to `orchestration/graph.py` and its routing edge |
| A model provider | Add it to the provider chain in `llm.py` |

Trip length, number of cities and travellers are parameters of the request, not of the code.

**Limits**: model and map quotas (free tiers run out; several Gemini keys spread the load), in-process caches that live until the server restarts, SQLite for storage, one plan planned at a time per request, and a 0.35 s solve per day.

## 5. Known limitations

- Hotel, flight, train and food prices, and the sights themselves, are model estimates, not bookable inventory.
- Results can differ between runs because the model and the forecast do.
- The visiting order is optimal only within 4,000 expansions; the day timetable is a good heuristic solution, not a proven optimum.
- Agent ownership of state is by convention, not enforced.
- There is no baseline comparison on this branch.
