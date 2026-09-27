# Odyssey — Multi-Agent Travel Planner (deterministic version)

> This is the `simple-deterministic` branch: **no LLM, no external API, no API keys**. Every decision is a rule
> or a search algorithm from the course, and the same request always gives the same plan. The LLM version is on `main`.

Odyssey turns a trip request ("5 days in Kerala with 3 friends, ₹40,000, nature and adventure, relaxed pace") into a
day-by-day itinerary using **six cooperating agents** that share one `TripState`. When a city closes, a storm or strike
hits, or the budget is cut, the Critic sends the plan back to **only the agent responsible**. After a plan exists you can
change it by typing ("make day 2 lighter", "add Varkala"), and the Edit Router re-runs only what the change touches.

```
User
  → 1. Trip Analyst          keyword / pattern parsing → TripSpec
  → 2. Destination Discovery cosine ranking + UCS proximity check
  → 3. Mobility & Routing    UCS on road/rail networks, A* (MST heuristic) for the visiting order
  → 4. Budget Optimization   tier rules + greedy repair (cheaper hotels, then least value-per-rupee sights)
  → 5. Itinerary Architect   CSP per day: backtracking + MRV + forward checking
  → 6. Critic & Replanner    rule checks → re-invoke exactly one agent (max 3 loops)
         ├─ valid → final itinerary
         ├─ closure / weather / far city / empty days → Destination
         ├─ strike / fares too high → Mobility
         ├─ over budget / budget cut → Budget
         └─ schedule conflict → Itinerary Architect
  7. Edit Router (for changes): rule grammar → re-enter at the earliest affected agent
```

## The agents

| # | Agent | Decides (by rule) | Algorithm | Must not |
|---|---|---|---|---|
| 1 | **Trip Analyst** | Region, days, travellers, budget, interests, pace, travel mode, start date; reports what it assumed; asks when the place is not covered | Keyword / regex parsing | Pick cities, hotels, a schedule |
| 2 | **Destination Discovery** | Which cities (named first; never closed / excluded / avoided; `ceil(days/2)` cities, max 4), sights per city | Cosine similarity of interests vs city profile; UCS road time ≤ daily limit between stops; best cluster over every seed city | Order cities, book hotels |
| 3 | **Mobility & Routing** | Road or rail for each hop (asked-for mode → cheapest if sent back over budget → rail if no slower or the drive is over the limit) | **UCS** on the road and rail networks; **A\*** over (city, visited set) with an admissible **MST heuristic**; UCS and greedy run alongside for comparison | Add cities, pick hotels |
| 4 | **Budget Optimization** | Hotel and food tier from budget per person; one tier cheaper after an overrun; traveller's hotel choice wins | Greedy repair: cheapest hotels first, then drop the sight with the least preference per rupee (fractional-knapsack greedy) | Reorder cities, schedule |
| 5 | **Itinerary Architect** | Nothing new: pace, free/light days, pins come from the request and edits | **CSP** per day (variables = sights + meals, 15-min domains, no-overlap-with-travel constraints), **backtracking + MRV + forward checking**; multi-objective score | Change cities or hotels |
| 6 | **Critic & Replanner** | Which one agent re-runs | Rule-based validation, severity-sorted routing, max 3 iterations, no repeat of a failed retry | Rewrite the plan, restart the Analyst |
| 7 | **Edit Router** | What a typed change means and where to re-enter | Rule grammar (days, pace, cities, sights, mode, hotels, length, budget, people, interests, events, questions) | Rewrite the plan itself |

The agents never call each other. They read and write the shared `TripState`, and LangGraph (orchestration only, no model)
runs them in order and follows the Critic's routing.

## Data

`backend/data/kerala.json` is the whole world the agents know: 8 destinations, 5 junction towns, a sparse **road network**
(16 links) and **rail network** (8 links), 47 sights with opening hours, prices, ratings and coordinates, 3 hotels per city
(budget / mid / premium), food prices per tier and **monthly weather averages** (hill stations cooler and wetter). A new
region is a new JSON file; no code changes.

## Run it

Prerequisites: Python 3.11+, Node.js 18+. No keys.

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000        # http://localhost:8000/docs

# frontend (second terminal)
cd frontend
npm install
npm run dev                                       # http://localhost:3000
```

## Tests and experiments

```bash
cd backend
pytest -q                         # 66 tests, under a second, fully offline
python -m scripts.compare_search  # A* vs UCS vs greedy on every subset of cities (Review 2 table)
```

The tests cover: UCS paths through junction towns; **A\* optimal on every 2–5 city subset (checked against brute force)**;
the MST heuristic never overestimating (checked against brute force); A\* expanding fewer nodes than UCS; the CSP respecting
opening hours, meals and travel time and dropping the least-wanted sight when a day is too full; every agent's rules; Critic
routing for each disruption; the edit grammar; and full scenarios through the HTTP API (normal plan, closure, strike, budget
cut, edits re-running from the right agent, a question leaving the plan alone, and determinism).

Search comparison (`scripts/compare_search.py`, average nodes expanded, road times):

| cities | A\* (MST) | A\* (min-edge) | UCS | greedy optimal |
|---|---|---|---|---|
| 4 | 9.8 | 16.3 | 26.7 | 70/70 |
| 5 | 15.9 | 44.4 | 67.8 | 49/56 |
| 6 | 25.1 | 115.5 | 164.6 | 18/28 |
| 7 | 36.9 | 282.6 | 383.6 | 2/8 |
| 8 | 48.0 | 649.0 | 854.0 | 0/1 |

All three A\*/UCS variants always find the optimum; the MST heuristic expands about 18× fewer nodes than UCS at 8 cities,
while greedy nearest-neighbour increasingly misses the optimum.

## Demo script (reviews)

1. **Plan**: "5 days in Kerala with 3 friends, around ₹40,000, nature and adventure at a relaxed pace". Watch the six agents;
   each shows the tools and algorithms it ran (A\* nodes vs UCS nodes, CSP nodes/backtracks). The Critic sends the first
   draft back to Budget once (over the ceiling), which goes one tier cheaper.
2. **Disruption**: click **Close \<city\>**. The Critic routes to Destination only; the other cities are kept and the closed
   one is replaced. Try **Cut budget** (→ Budget) and **Transport** (→ Mobility).
3. **Edit by typing**: "make day 2 lighter" (→ Architect), "by train" (→ Mobility), "add Alleppey" (→ Destination),
   "how much does it cost?" (answered, plan unchanged).

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/plan` | Start a planning session → `{thread_id}` |
| `GET` | `/api/plan/{thread_id}/stream` | SSE stream of the run (`step_start`, `step_complete` with the agent's tools/algorithms, `done`, `error`) |
| `POST` | `/api/disrupt` | Inject a closure / weather / transport / budget-cut event, then reconnect to the stream |
| `POST` | `/api/revise` | Change the plan with a sentence, then reconnect to the stream |
| `GET` | `/api/itinerary/{thread_id}` | Current itinerary (kept in memory while the server runs) |
| `GET` | `/api/health` | Status and the regions loaded |

## Known limitations

- One region (Kerala) is bundled. Other places get a question back, not a plan.
- Plans are kept in memory: restarting the backend forgets them.
- Weather is a monthly average, not a forecast. Prices and times are fixed dataset values.
- The request and edit parsers understand the patterns listed in the agent files; unusual phrasing gets a "here is what I
  can change" reply.
- Joint optimality is not guaranteed across agents: each agent optimises its own part (order, schedule, budget) given the
  earlier agents' output; the Critic's loop repairs, it does not search the whole space.

## Repository layout

```
backend/data/          the offline world (one JSON per region)
backend/algorithms/    search.py (UCS), astar.py (A*, MST heuristic, greedy), csp_solver.py (CSP), optimizer.py, planning.py
backend/agents/        the six agents + edit router
backend/orchestration/ LangGraph state, graph and routing
backend/tools/         world.py (dataset, UCS distances, weather), costs, scoring, validators
backend/scripts/       compare_search.py (the search comparison table)
frontend/              Next.js 14 + Tailwind UI
```
