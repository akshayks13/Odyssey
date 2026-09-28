# Odyssey — Review 2: Implementation

Branch `simple-deterministic` (rule-based agents, hand-written search, no LLM, no external API). The LLM version is on `main`.

Rubric lines covered: **Tool/package selection and setup** (§1) · **Multi-agent execution and interaction** (§2) · **Demo quality and testing scenarios** (§3) · **Code structure and scalability** (§4). The design behind this is in [`review1_design_report.md`](review1_design_report.md); the full tables are in [`PLAN.md`](../PLAN.md).

## 1. Tool / package selection and setup

### What is used and why

| Tool | Version | Used for | Why this one |
|---|---|---|---|
| Python | 3.11+ (tested on 3.13) | The whole backend | Course language |
| `heapq` (standard library) | — | The priority queue for uniform-cost search and A\* | The searches and the CSP are hand-written because they are the graded part |
| Pydantic | 2.13.5 | Validated schemas for every piece of state | A malformed value is rejected at the boundary |
| FastAPI + uvicorn | 0.115.6 / 0.32.1 | HTTP API and server-sent events | Streams every message and every agent step to the UI as it happens |
| httpx | 0.28.1 | The test client for the API | Required by FastAPI's `TestClient` |
| pytest | 8.3.4 | 126 tests | One command runs everything, offline, in about two seconds |
| matplotlib | 3.11.2 (`requirements-dev.txt`) | The experiment charts | Optional: the tables need nothing extra |
| Next.js 14, Tailwind, MapLibre GL, Recharts | see `frontend/package.json` | The web UI: agent timeline, live message log, day-by-day plan, budget chart, route map, strategy comparison | The UI needs the streamed events and the plan |

There is no model, no key and no network call: the world is one file, `backend/data/kerala.json` (8 cities, a road network of 16 links and a rail network of 8, 47 sights with hours, prices and coordinates, 24 hotels, food prices, monthly weather).

### Considered and not used

| Option | Why not |
|---|---|
| An LLM | Non-deterministic, needs keys and quota, and hides the reasoning the course grades |
| LangGraph or another agent framework | The message bus and the ownership rules *are* the multi-agent design; a framework would hide them |
| OR-Tools, NetworkX | They would do the search and the CSP we are meant to write |
| Live travel and weather APIs | Quota, and results change between runs |
| Mesa | Built for grid simulations; the world here is a graph |
| SQLite persistence | Not needed: plans are kept in memory while the server runs |

### Setup

Requires Python 3.11+ and Node 18+. No keys.

```bash
./setup.sh                 # creates backend/.venv, installs the Python and npm packages
./run.sh                   # backend on :8000 and the UI on http://localhost:3000
./run.sh tests             # the test suite
./run.sh experiments       # the strategy comparison (add --quick for 3 seeds)
```

By hand:

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # requirements.txt is enough to run and test
uvicorn api.main:app --reload --port 8000

cd frontend && npm install && npm run dev
```

## 2. Multi-agent execution and interaction

### Messages

The agents never call each other. Every arrow is one of 15 message types on a bus (`backend/core/messages.py`):

| Message | From → To |
|---|---|
| `REQUEST` | traveller → Trip Analyst |
| `SPEC_READY` / `NEED_INFO` | Trip Analyst → Destination / traveller |
| `CITIES_READY` | Destination → Mobility |
| `ROUTE_READY` | Mobility → Budget |
| `BUDGET_READY` | Budget → Architect |
| `SCHEDULE_READY` | Architect → Environment (or straight to the Critic when the strategy does not check the field) |
| `FIELD_REPORT` | Environment → Critic |
| `REPLAN` | Critic → the one agent that owns the issue |
| `ACCEPT` / `BEST_EFFORT` | Critic → traveller |
| `DISRUPTION` | traveller → Critic |
| `EDIT` / `RERUN` / `ANSWER` | traveller → Edit Router → the earliest affected agent, or the traveller |

A real run: *5 days in Kerala, ₹60,000, from 10 July, nature and adventure, relaxed*, on a rough field (seed 3).

| # | From → To | Message | What it says |
|---|---|---|---|
| 1 | traveller → Analyst | `REQUEST` | the sentence |
| 2 | Analyst → Destination | `SPEC_READY` | 5 days, ₹60,000, 4 people |
| 3 | Destination → Mobility | `CITIES_READY` | Thekkady, Vagamon, Munnar |
| 4 | Mobility → Budget | `ROUTE_READY` | Vagamon → Thekkady → Munnar, 4.5 h, ₹1,620 |
| 5 | Budget → Architect | `BUDGET_READY` | ₹51,260 of ₹60,000 (within budget) |
| 6 | Architect → Environment | `SCHEDULE_READY` | 5 days, 5 with sights, score 0.81 |
| 7 | Environment → Critic | `FIELD_REPORT` | 18 checked: heavy rain in Thekkady on day 3; heavy rain in Munnar on day 4 … |
| 8 | Critic → Architect | `REPLAN` | HEAVY_RAIN: outdoor sights are scheduled on a day of heavy rain |
| 9 | Architect → Environment | `SCHEDULE_READY` | 5 days, 4 with sights, score 0.69 |
| 10 | Environment → Critic | `FIELD_REPORT` | 2 checked: everything as expected |
| 11 | Critic → traveller | `ACCEPT` | plan accepted |

Only the Architect re-ran (message 8): Destination, Mobility and Budget each ran once.

### Ownership of state

All agents share one blackboard, but each declares the keys it `reads` and `writes`. It is handed a view containing only its reads (asking for anything else raises `AccessError`), and its updates are refused if they touch a key it does not own. The one key two agents may write is the activity line of the budget, which the Architect reconciles to what it really scheduled. The test suite runs whole sessions of plans, events and every kind of edit under this enforcement.

### Who fixes what (the Critic)

| Issue | Sent to | What that agent does differently |
|---|---|---|
| Closure or storm reported at a city | Destination | Keeps the cities that are fine, replaces that one |
| A hop over the daily limit | Destination | Avoids the far city (unless the traveller named both) |
| A day with nothing left to see | Destination | Takes one more city |
| A strike | Mobility | Slower road for a reported strike; the train, where one runs, for a one-day strike |
| Over budget, fares ≥ 40% of it | Mobility | Cheapest mode per hop |
| Over budget | Budget | One tier cheaper |
| Budget cut | Budget | Re-prices against the new ceiling |
| Overlap in a schedule | Architect | Re-solves the day |
| Heavy rain reported on a day with outdoor sights | Architect | Drops the outdoor sights from that day; they stay available for later days |
| A scheduled sight reported closed | Architect | Drops it for that day |

An issue still present after its agent retried is not sent again (a field issue is, when it is a new fact); at most five loops; then the best plan so far is returned with what is left.

### Conflicts

| Conflict | Resolution |
|---|---|
| Destination's favourite city vs Budget's overrun | Cheapest-damage repair first (hotel tier or travel mode), before dropping the city |
| Budget priced sights before they were scheduled | The Architect reconciles the activity line to what it scheduled |
| A closure vs a traveller's pinned city | The closure wins; the pin is removed and the city excluded |
| A field report vs the monthly average | The report wins for that day |

Standing choices (`EditLocks`: pinned and excluded cities, mode, hotels, skipped and pinned sights, free and light days, pace, start time) are read by every agent, so a later replan never undoes an edit.

### Strategies (each switches off one design choice)

| Strategy | What it changes |
|---|---|
| `full` | The system: A\* order, CSP timetable, targeted replanning, field checks |
| `greedy_order` | Nearest-neighbour order instead of A\* |
| `greedy_schedule` | First-fit timetable instead of the CSP |
| `restart` | Any issue re-runs every agent from Destination |
| `static` | Plans once; no field checks, no replanning |

## 3. Demo quality and testing scenarios

### Demo script

1. **Plan, in a calm world.** Field: *Calm world*. "5 days in Kerala with 3 friends, around ₹40,000, nature and adventure at a relaxed pace". Each timeline row shows the tools and algorithms that agent ran (A\* nodes against uniform-cost nodes, CSP nodes and backtracks). The *Messages between agents* log shows every hand-off. The Critic sends the first draft back to Budget (over the ceiling), which goes one tier cheaper.
2. **The world surprises the plan.** Field: *Rough*, and a rainy month ("… from 2027-07-10"). A *Field check* row appears after the Schedule; the environment reports heavy rain on some days, the Critic sends only the Architect back, and the outdoor sights move to dry days.
3. **Events.** *Close ⟨city⟩* → Destination only, the other cities stay. *Cut budget* → Budget. *Transport issue* → Mobility.
4. **Change it by typing.** "make day 2 lighter" (→ Architect), "by train" (→ Mobility), "add Alleppey" (→ Destination), "how much does it cost?" (answered, plan unchanged).
5. **Does each part earn its place?** *Compare strategies on this trip* plans the same trip five ways and scores each against what really happens.

### The test suite

`./run.sh tests` (or `cd backend && pytest -q`): **126 tests, offline, about two seconds.**

| File | Tests | What it checks |
|---|---|---|
| `test_algorithms.py` | 20 | UCS through junction towns; A\* equals the brute-force optimum on every 2–5 city subset; the MST heuristic never overestimates; the CSP's windows, meals and travel; the first-fit baseline loses a sight the CSP keeps |
| `test_agents.py` | 43 | The parser; each agent's rules; the Critic's routing; the strategy variants; the edit grammar |
| `test_team.py` | 25 | The message chain; ownership of state; the field is reproducible and its frequencies match the rates; the team learns only what the plan uses; heavy rain replans only the Architect; strategies differ as claimed |
| `test_scenarios.py` | 21 | Whole scenarios through the HTTP API, the path the UI takes |
| `test_experiments.py` | 11 | Every strategy on every seed, paired statistics, reproducible output |
| `test_docs.py` | 6 | `PLAN.md` and this report contain every agent's PEAS |

### Scenarios

| # | Scenario | Expected | Test |
|---|---|---|---|
| 1 | Normal request | Six agents, valid, within budget | `test_normal_plan_runs_all_six_agents_and_is_valid` |
| 2 | Same request twice | Identical plans, also with the field on | `test_same_request_gives_the_same_plan`, `test_same_request_meets_the_same_field_…` |
| 3 | A place not in the data | A question, no plan | `test_unknown_region_asks_instead_of_guessing` |
| 4 | A city is reported closed | Critic → Destination only; other cities stay | `test_closing_a_city_replans_from_destination_only` |
| 5 | The budget is cut | Critic → Budget; new ceiling | `test_budget_cut_reprices_from_budget` |
| 6 | A strike | Critic → Mobility | `test_strike_replans_the_route` |
| 7 | Heavy rain is found in the field | Only the Architect replans; the result survives the real weather | `test_heavy_rain_is_found_and_only_the_architect_replans` |
| 8 | Edits | Re-run from the earliest affected agent only | `test_edits_rerun_only_from_the_earliest_affected_agent` |
| 9 | A question | Answered; plan unchanged | `test_a_question_leaves_the_plan_alone` |
| 10 | An agent writes or reads outside its declaration | `AccessError` | `test_an_agent_may_not_write_…`, `test_an_agent_sees_only_…` |
| 11 | Plan once vs the full system | The full system saves sights that plan-once loses | `test_the_static_strategy_loses_sights_the_full_system_saves` |
| 12 | Restart vs targeted replanning | Same quality, more agent runs | `test_restarting_costs_more_agent_runs_than_a_targeted_replan` |

### Experiments

`./run.sh experiments`: 9 scenarios × 5 strategies × 30 seeds = 1,350 runs. For every (scenario, seed) all five strategies plan the same trip, meet the same field and hear the same reported events. **Value** is the share of the no-surprise plan's value that survives the field. Charts and CSVs are in `backend/results/`.

| Scenario | Full system | Greedy order | Greedy timetable | Restart | Plan once |
|---|---|---|---|---|---|
| calm | 96.5 | 95.0 | 65.3 | 96.5 | 91.2 |
| monsoon | 88.5 | 86.5 | 60.4 | 88.5 | 82.3 |
| closures | 90.8 | 88.6 | 64.5 | 90.8 | 71.2 |
| strikes | 77.6 | 74.8 | 54.1 | 74.3 | 66.1 |
| budget_cut | 97.0 | 96.9 | 69.3 | 97.7 | 73.3 |
| scale_3 | 83.6 | 84.3 | 50.6 | 83.6 | 76.9 |
| scale_5 | 90.4 | 88.7 | 60.5 | 90.4 | 84.7 |
| scale_8 | 95.8 | 95.8 | 67.6 | 95.8 | 78.2 |
| scale_12 | 89.4 | 91.8 | 67.3 | 80.2 | 69.8 |

What it shows (paired on the same seed, 95% intervals in `PLAN.md` §9):
- Checking the field and repairing beats planning once in all nine scenarios (+5 to +24 points, significant each time).
- The CSP beats a first-fit timetable by 22 to 33 points everywhere.
- Restarting reaches the same value in most scenarios but needs 1.5 to 4.7 more agent runs per trip.
- A\* over nearest neighbour barely shows end to end: trips visit at most six places, and nearest neighbour is optimal for most small sets. It shows in the search comparison (48 against 854 nodes at eight cities; nearest neighbour stops being optimal from five cities).
- Some losses cannot be fixed: in `strikes` the full system still delivers 78%, because a strike on a day that must be driven, where no train runs, cannot be avoided.

The first-fit baseline is deliberately simple; a smarter greedy heuristic would narrow the CSP's gap.

## 4. Code structure and scalability

```
backend/
  core/         messages.py (the protocol), blackboard.py (ownership), team.py (the runtime), environment.py (the field),
                strategies.py, metrics.py, compare.py
  agents/       one module per agent: a class with role, PEAS, reads, writes (+ base.py)
  algorithms/   search.py (UCS), astar.py (A*, MST heuristic, greedy), csp_solver.py (CSP and first-fit), planning.py, optimizer.py
  tools/        world.py (the dataset, UCS travel times, weather), costs, scoring, validators
  data/         kerala.json: the whole world
  experiments/  scenarios.py, run_experiments.py, search_comparison.py
  results/      the CSVs and charts the experiments wrote
  api/          FastAPI routes and the event stream
  tests/
frontend/       Next.js UI
PLAN.md  README.md  docs/
```

About 4,200 lines of Python plus 1,050 of tests. Algorithms are pure functions with no I/O and are tested directly; the runtime (`core/`) makes no decisions, every decision is an agent's.

| To add | What to do |
|---|---|
| A region | A new JSON file in `backend/data/`, no code change |
| An agent | A class in `agents/` with its `peas`, `reads`, `writes` and `handle`; add it to `Team` |
| A strategy | One line in `core/strategies.py`; the experiments and the UI comparison pick it up |
| A scenario | One entry in `experiments/scenarios.py` |

Trips from 1 to 30 days and up to six stops run without code changes; a 12-day trip plans in under 15 milliseconds (median 7 ms over 15 seeds).

**Limits.** One region. Weather is a monthly average and the surprises are a seeded model. Rule-based parsing means unusual phrasing gets a "here is what I can change" reply. Plans live in memory, so restarting the backend forgets them. Each agent optimises its own part, so the joint plan is not globally optimal.
