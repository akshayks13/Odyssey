# Odyssey — design (deterministic version)

Branch `simple-deterministic`. Six rule-based agents and course search algorithms; no language model, no external API.
How to run it is in [`README.md`](README.md). The LLM design is on `main`.

## 1. Problem

Given a traveller's request (region, days, people, budget, interests, pace), produce a day-by-day itinerary that visits
cities they will like, in a sensible order, within budget, with every sight inside its opening hours. When the world changes
(a city closes, a storm, a strike, a budget cut) or the traveller asks for a change, repair only the part that is affected.

## 2. PEAS (Review 1)

| | |
|---|---|
| **Performance** | Feasible plan (no overlaps, every sight inside its hours, meals in their windows); total ≤ budget; no hop longer than the daily travel limit; high interest match; minimum travel time between cities; multi-objective score `S = w_p·P + w_q·Q + w_r·R + w_b·B − w_t·T − w_c·C` (weights by traveller archetype); few replan loops |
| **Environment** | The Kerala dataset (cities, road and rail networks, sights, hotels, prices, monthly weather), the traveller, and events injected while the plan exists (closures, storms, strikes, budget cuts, typed edits) |
| **Actuators** | Write their part of the shared `TripState` (spec, cities, route, budget, schedule), route a replan to one agent, ask the traveller a question, answer a question |
| **Sensors** | The request text, the dataset lookups (`road_route`, `rail_route`, `sights`, `hotels`, `weather_on`), disruption events, edit messages, the Critic's validation report |

## 3. Environment and agent analysis (Review 1)

| Property | Value | Why |
|---|---|---|
| Observable | **Fully** (the dataset); events are unknown in advance | Every agent can read all data it needs; what it cannot see is the future (a closure) |
| Deterministic | **Yes** | An action always has the same effect; same request → same plan (a test checks this) |
| Episodic / sequential | **Sequential** | Destination's choice changes Mobility's graph, Mobility's fares change Budget, Budget's cuts change the schedule |
| Static / dynamic | **Dynamic** between runs, static during one run | Closures, strikes, budget cuts and edits arrive while the plan exists and trigger replanning |
| Discrete / continuous | **Discrete** | Cities and towns are graph nodes; time is 15-minute slots; tiers are three values |
| Agents | **Multi-agent, cooperative** | Six specialists with separate goals (interest, travel time, money, feasibility) sharing one state; the Critic coordinates |
| Known | **Known** | Travel times, prices and rules are given in the dataset |

Agent types: the Trip Analyst and Edit Router are **simple reflex** agents (condition–action rules on the text).
Destination, Mobility and the Architect are **goal-based** (they search for an assignment that satisfies a goal: a cluster
of close cities, an optimal visiting order, a consistent timetable). Budget and the Critic are **utility-based**: Budget
repairs with the least damage per rupee saved, and the Critic picks the most severe issue and the cheapest agent to fix it.
The system as a whole maximises the score `S`.

## 4. Algorithms (Review 1: modelling and search strategy)

### 4.1 Uniform-cost search — travel time between two towns (`algorithms/search.py`)

- **State**: a town. **Actions**: take one road (or rail) link. **Step cost**: hours for that link (> 0).
- **Goal**: the destination town. Frontier ordered by path cost `g(n)`; goal tested when a node is expanded.
- Complete and optimal because every step cost is positive. The road network is sparse and includes junction towns, so
  Kochi → Wayanad is found as Kochi → Thrissur → Kozhikode → Wayanad (7.5 h).

### 4.2 A\* — the order to visit the chosen cities (`algorithms/astar.py`)

- **State**: `(current city or None, frozenset of visited cities)`; **initial**: `(None, ∅)`.
- **Actions**: go to any unvisited city. **Step cost**: hours between the two cities (from 4.1, by the chosen mode); 0 for
  the first city. **Goal**: all cities visited. This is an open travelling-salesman path.
- **Heuristic**: weight of a **minimum spanning tree** over the current city and the unvisited ones (Prim).
  *Admissible*: any way to finish is a path through those cities, and a path is a spanning tree, so it costs at least the
  MST. *Consistent*: adding the edge current→next to the next state's MST gives a spanning tree of this state.
  So A\* with a closed set returns an optimal order.
- **Baselines on the same problem**: UCS (`h = 0`), A\* with the weaker "cheapest edge out" heuristic, and greedy nearest
  neighbour. `scripts/compare_search.py` prints the table in the README: at 8 cities A\*-MST expands 48 nodes, UCS 854, and
  greedy is optimal in 0 of 1 cases.

### 4.3 CSP — each day's timetable (`algorithms/csp_solver.py`)

- **Variables**: each chosen sight, plus lunch and dinner.
- **Domains**: start times in 15-minute steps, after the sight opens and after the traveller can reach it from the hotel,
  and early enough to finish before it closes and before the day ends (lunch 12–14, dinner 19–21).
- **Constraints** (binary, every pair): `start(Y) ≥ end(X) + travel(X, Y)` or `start(X) ≥ end(Y) + travel(Y, X)`.
- **Search**: backtracking; **MRV** chooses the variable with the fewest values left; values tried earliest first (compact
  day); **forward checking** removes clashing values from every other domain after each assignment and backtracks as soon
  as one is empty. If the day cannot hold every sight, the least-wanted one is dropped and the CSP is solved again.

### 4.4 Other decision rules

- **Interest match**: cosine similarity between the traveller's 7-theme vector and the city's profile.
- **Budget repair** (greedy, cheapest damage first): switch unchosen hotels to the cheapest; then drop the paid sight with
  the lowest preference per rupee (the fractional-knapsack greedy rule), keeping at least one sight per city.
- **Score**: `S = w_p P + w_q Q + w_r R + w_b B − w_t T − w_c C`, weights by archetype (adventure / relaxed / budget / balanced).

## 5. Multi-agent interaction (Review 2)

```
START ─┬─ new plan → Trip Analyst ─┬─ unknown place → ask the traveller (END)
       │                           └─ Destination → Mobility → Budget → Architect → Critic ─┬─ valid → END
       │                                                                                    ├─ Destination
       │                                                                                    ├─ Mobility
       │                                                                                    ├─ Budget
       │                                                                                    └─ Architect
       └─ edit → Edit Router ─→ the earliest agent whose inputs changed (or answer → END)
```

- **Shared state, no direct calls**: each agent reads what earlier agents wrote in `TripState` and writes only its own part.
- **Critic routing** (one agent per loop, at most 3 loops, a reported event first, then by severity):

  | Issue | Sent to | What that agent does differently |
  |---|---|---|
  | CLOSURE / WEATHER at a city | Destination | keeps the other cities, replaces that one |
  | TRAVEL_OVERLOAD (hop > daily limit) | Destination | avoids the far city (unless the traveller named both: then only reported) |
  | LEISURE_DAY (nothing left to see) | Destination | adds one more city |
  | TRANSPORT (strike) | Mobility | road in/out of that city is 2× slower, 1.5× dearer; rail preferred |
  | BUDGET, fares ≥ 40% of the budget | Mobility | cheapest mode per hop |
  | BUDGET | Budget | one tier cheaper |
  | BUDGET_CUT | Budget | re-prices against the new ceiling |
  | SCHEDULE_CONFLICT | Architect | re-solves the day |

- An issue still present after its agent already retried is not sent again; the plan is returned with the issue listed.
- **Conflicts** are recorded (`AgentConflict`), e.g. Destination's top city vs Budget's overrun, and resolved by the Critic
  choosing the cheapest-damage repair (hotel tier or travel mode) before dropping a city.
- **Standing decisions** (`EditLocks`): pinned / excluded cities, travel mode, hotel choice, skipped sights, pinned sights,
  free and light days, pace, start time. Every agent reads them, so a later replan never undoes an edit. Handled closures and
  budget cuts are folded into them too.

## 6. Tools and packages (Review 2)

| Tool | Why |
|---|---|
| Python 3.11+ | Course language |
| `heapq` (stdlib) | Priority queue for UCS and A\*; the searches themselves are hand-written because they are the graded part |
| LangGraph `StateGraph` | Orchestration only: shared typed state, conditional edges for the Critic's routing, in-memory checkpoints so a disruption or edit resumes the same plan. No model |
| Pydantic | Validated schemas for every piece of state |
| FastAPI + SSE | Streams each agent's step (message, tools, algorithms) to the UI as it runs |
| Next.js + Tailwind + MapLibre | Web UI: agent timeline, day-by-day plan, budget chart, route map |
| pytest | 66 tests, offline, under a second |

Considered and not used: an LLM (non-deterministic, needs keys and quota, and hides the reasoning the course grades);
OR-Tools and NetworkX (they would hide the search and CSP we are meant to implement); live travel APIs (quota, and results
change between runs).

## 7. Testing scenarios (Review 2)

| # | Scenario | Expected | Test |
|---|---|---|---|
| 1 | Two towns with no direct road | UCS finds the path through junctions | `test_ucs_finds_the_path_through_junction_towns` |
| 2 | Every 2–5 city subset | A\* order = brute-force optimum | `test_astar_is_optimal_on_every_subset` |
| 3 | Every partial state | MST heuristic ≤ true remaining cost | `test_mst_heuristic_never_overestimates` |
| 4 | Six cities | A\* expands fewer nodes than UCS; greedy ≥ optimum | `test_astar_expands_fewer_nodes_than_ucs_and_beats_greedy_or_ties` |
| 5 | Day with opening hours, meals, travel | No overlap, all windows kept | `test_csp_respects_opening_hours_meals_and_travel` |
| 6 | Two sights needing the same morning | Least-wanted dropped | `test_csp_drops_the_least_wanted_sight_when_the_day_cannot_hold_all` |
| 7 | Normal request | 6 agents, valid, within budget | `test_normal_plan_runs_all_six_agents_and_is_valid` |
| 8 | Same request twice | Identical plans | `test_same_request_gives_the_same_plan` |
| 9 | Place not in the dataset | A question, no plan | `test_unknown_region_asks_instead_of_guessing` |
| 10 | City closed | Critic → Destination only; other cities kept | `test_closing_a_city_replans_from_destination_only` |
| 11 | Budget cut | Critic → Budget; new ceiling | `test_budget_cut_reprices_from_budget` |
| 12 | Strike | Critic → Mobility | `test_strike_replans_the_route` |
| 13 | Edits | Re-run from the earliest affected agent only | `test_edits_rerun_only_from_the_earliest_affected_agent` |
| 14 | A question | Answered, plan unchanged | `test_a_question_leaves_the_plan_alone` |

## 8. Viva one-liners

- **Is the A\* heuristic admissible?** Yes: finishing the trip is a path through the remaining cities, a path is a spanning
  tree, so it costs at least the MST.
- **Why UCS for city-to-city times?** Positive step costs, so the first time the goal is expanded its cost is the shortest.
  Goal test on expansion, not generation.
- **Why not greedy for the order?** It is fast but not optimal: at 7 cities it finds the optimum in 2 of 8 subsets.
- **Why is the schedule a CSP?** Every sight has a time window and no two can overlap once travel is counted; that is
  variables, domains and binary constraints.
- **What do MRV and forward checking buy?** MRV places the most constrained sight first (e.g. a show that only runs 17–20);
  forward checking detects a dead end one step early instead of at the bottom of the tree.
- **How is the replan targeted?** The Critic maps each issue type to the one agent that owns it, and LangGraph re-enters
  the pipeline there; the agents before it are not re-run.
- **Why is it multi-agent and not one program?** Each agent has its own goal and knowledge (interests, travel time, money,
  timetable), acts only on its part of the shared state, and the Critic coordinates their conflicts.
- **Limitations?** One region; weather is a monthly average; each agent optimises its own part given earlier agents' output,
  so the joint plan is not guaranteed globally optimal.
