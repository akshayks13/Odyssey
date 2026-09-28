# Odyssey — Review 1: Design Report

Branch `simple-deterministic` (rule-based agents, hand-written search, no LLM, no external API). The LLM version is on `main`.

Rubric lines covered: **PEAS formulation** (§3) · **Environment and agent analysis** (§4–5) · **Algorithmic modelling and search strategy** (§6–7) · **Q&A** (§8). The full design, with the coordination tables and results, is in [`PLAN.md`](../PLAN.md).

## 1. Introduction

**Problem.** A traveller types a trip ("5 days in Kerala with 3 friends, ₹60,000, nature and adventure, relaxed pace"). The team must return a day-by-day itinerary: cities they will like, in a sensible order, within budget, with every sight inside its opening hours. But the plan is made on **averages** (a month's typical weather) and the world is not average: on some days it rains hard, a sight is shut, a road strike hits a travel day, the traveller cuts the budget. The agents only find out what is real by **checking**, and when they do, only the agent that owns the problem should redo its work.

**Objectives**
1. Model the trip as a partially observable, stochastic, dynamic environment with a known, seeded model of surprises.
2. Build agents with distinct roles that share no code paths: each owns part of a shared state and they talk through explicit messages.
3. Use informed search (A\*) for the visiting order, uniform-cost search for travel times, and a constraint satisfaction solver for each day's timetable.
4. Show by controlled experiments over many seeds that each design choice earns its place, and say where it does not.

**Scope.** One region (Kerala), modelled by hand in `backend/data/kerala.json`. Prices, times and weather averages are approximate; hotel names are placeholders.

## 2. System overview

```
traveller ─REQUEST─► Trip Analyst ─SPEC_READY─► Destination ─CITIES_READY─► Mobility ─ROUTE_READY─► Budget
                                                                                                      │ BUDGET_READY
traveller ◄─ACCEPT / BEST_EFFORT─ Critic ◄─FIELD_REPORT─ Environment ◄─SCHEDULE_READY─ Itinerary Architect
                  │  ▲
                  │  └── DISRUPTION ── traveller
                  └── REPLAN ──► the ONE agent that owns the issue (Destination, Mobility, Budget or Architect)

traveller ─EDIT─► Edit Router ─RERUN─► the earliest agent whose inputs changed   (or ANSWER ► traveller)
```

| # | Agent | Role | How it decides |
|---|---|---|---|
| 1 | Trip Analyst | Parser | Keyword and pattern rules |
| 2 | Destination Discovery | Explorer | Cosine interest match; uniform-cost search for "close enough" |
| 3 | Mobility & Routing | Mover | Uniform-cost search on road and rail networks; A\* for the order |
| 4 | Budget Optimization | Money | Tier rules; greedy repair, cheapest damage first |
| 5 | Itinerary Architect | Scheduler | CSP per day: backtracking, MRV, forward checking |
| 6 | Critic & Replanner | Coordinator | Rule-based validation; sends each issue to one owner |
| 7 | Edit Router | Front door for changes | Rule grammar over typed sentences |

The **Environment** is a sensor, not an agent: it has no goal and answers only what it is asked.

## 3. PEAS formulation

**System performance measure** (shared by all agents): `S = w_p·P + w_q·Q + w_r·R + w_b·B − w_t·T − w_c·C`, where P = interest match of the scheduled sights, Q = their rating, R = route efficiency, B = budget efficiency, T = travel burden, C = constraint violations. Weights depend on the traveller's archetype (adventure, relaxed, budget, balanced).

The text below is the `peas` declared on each agent class in `backend/agents/` (`GET /api/agents` returns the same), and a test checks it matches.

| Agent | Performance | Environment | Actuators | Sensors |
|---|---|---|---|---|
| **Trip Analyst** | Every stated fact read correctly (days, people, budget, dates, interests, pace, travel mode); every unstated fact reported as an assumption; never a guessed place | The traveller's sentence and the list of regions the dataset covers | Write the trip spec; ask the traveller a question (NEED_INFO); send SPEC_READY | The request text; the region and city index |
| **Destination Discovery** | High interest match; every stop within a day's travel of another; no closed, avoided or storm-hit city; enough sights for the days spent there | The region's cities, their interest profiles, the road network, monthly weather, and the traveller's standing choices | Write the ranked cities, the chosen stops and their sights; send CITIES_READY | The trip spec, the edit locks, reported disruptions, the Critic's REPLAN directive |
| **Mobility & Routing** | Minimum total travel time between cities; no hop longer than the daily limit unless asked for; the mode the traveller asked for; fares that leave room for the rest of the budget | The road and rail networks (with junction towns), fares, transport strikes, and the traveller's travel-mode choice | Write the route (order, legs, mode, cost); send ROUTE_READY | The chosen cities, the trip spec, the edit locks, reported strikes, the Critic's REPLAN directive |
| **Budget Optimization** | Total at or under the ceiling; the fewest and least valuable sights dropped; the traveller's hotel choices kept | Hotel prices per tier, food and local-transport prices, fares from Mobility, the ceiling and any reported budget cut | Write the budget breakdown, the hotels, the stay plan and the sights dropped; send BUDGET_READY | The route and its fares, the candidate sights, the trip spec, the edit locks, reported budget cuts, the previous breakdown |
| **Itinerary Architect** | Every sight inside its opening hours and the day's window; no overlaps once travel time is counted; meals in their windows; the traveller's free, light and pinned days kept; high score | Opening hours, distances inside a city, the stay plan, the weather (expected and reported), sights reported closed | Write the day-by-day schedule and the reconciled cost of what it scheduled; send SCHEDULE_READY | The route, the sights and their hours, the budget, the stay plan, the edit locks, the field reports |
| **Critic & Replanner** | Every real problem caught; each sent to exactly one owner; the fewest replan loops; a valid plan accepted; the best plan so far returned when no retry can help | The draft plan, the budget, reported events, the field report, and the other agents' declared ownership | Write the validation report; send REPLAN to one agent, or ACCEPT / BEST_EFFORT to the traveller; fold handled events into standing choices | Everything on the blackboard (read only), incoming SCHEDULE_READY, FIELD_REPORT and DISRUPTION messages |
| **Edit Router** | Every change understood and stored as a standing choice; the pipeline re-entered no later than the earliest agent whose inputs changed; questions answered without touching the plan | The current plan, the traveller's standing choices, and one typed sentence | Update the trip spec and the edit locks; send RERUN to one agent, or ANSWER to the traveller | The plan (itinerary, budget, route, cities, sights) and the EDIT message |

## 4. Environment analysis

| Property | Value | Why | What it forces in the design |
|---|---|---|---|
| Observable | **Partially** | The dataset (roads, hours, prices, averages) is known. What is real on a given day is not: heavy rain, a closed sight, a strike. The environment answers only for the days, sights and journeys in the plan it is shown | Plan on averages, then check; keep what has been learned; count the checks |
| Deterministic | **Stochastic, but reproducible** | Surprises are seeded draws (a hash of seed, kind and key): the same seed always gives the same world | A plan is repaired, not assumed final; every strategy in an experiment meets the same field, so results are paired |
| Sequential | **Yes** | Cities change the route, the route's fares change the budget, the budget's cuts change the schedule | An order of work; a repair may re-run what follows it |
| Static / dynamic | **Dynamic** | Events arrive after the plan exists: reported closures, storms, strikes, budget cuts, typed edits | A Critic that reacts, and targeted replanning so a small event costs a small amount of work |
| Discrete | **Yes** | Towns are graph nodes, time is 15-minute slots, tiers are three values | Search over finite spaces is well defined |
| Agents | **Multi-agent, cooperative** | Six agents with different goals (interest, travel time, money, feasibility) and one shared score | Ownership of state, a message protocol, a coordinator for conflicts |
| Rules | **Known rules, unknown state** | The model of surprises is known; which draw comes up is not | Expected values guide the plan; observation corrects it |

**The field** (`backend/core/environment.py`). Heavy rain: each day with probability *(monthly rain chance × 0.5 × level)*, stopping boating, treks and viewpoints. A sight closed: 5% per sight per day. A road strike: 4% per city per day, felt on a day that drives in or out of it. Levels `off`, `normal`, `high`, and `closures` / `strikes` (one kind turned up). The measured frequencies are checked by a test (July rain about 42%, January about 5%).

**Sensing has a cost.** When the Architect hands over a schedule the environment looks up only the days, sights and journeys in it. Each fact is asked once and remembered; the number asked is reported as *field checks*.

## 5. Agent analysis

A simple reflex agent (condition → action, no memory) is enough to *parse text*: the Trip Analyst and the Edit Router. It is not enough for the rest, because the right action depends on state that is not in the current percept: a sight's opening hours against the day already built, fares already committed, what the field has already reported.

| Agent | Type | Why |
|---|---|---|
| Trip Analyst, Edit Router | Simple reflex | Condition–action rules on a sentence |
| Destination Discovery | Goal-based | Finds a set of stops meeting the interest and distance goals |
| Mobility & Routing | Goal-based | Searches for the fastest complete visiting order |
| Itinerary Architect | Goal-based | Finds a consistent timetable: a constraint satisfaction problem |
| Budget Optimization | Utility-based | Trades comfort for money by the cheapest damage per rupee saved |
| Critic & Replanner | Utility-based | Picks the most severe issue and the cheapest agent to fix it |

## 6. Algorithmic modelling

### 6.1 Uniform-cost search — travel time between two towns (`algorithms/search.py`)

| Element | Definition |
|---|---|
| State | A town |
| Actions | Take one road (or rail) link |
| Step cost | Hours for that link, always positive |
| Goal test | The destination town, tested when a node is *expanded* |

The frontier is ordered by path cost `g(n)`. Complete and optimal because every step cost is positive. The networks are sparse and include junction towns, so Kochi → Wayanad is found as Kochi → Thrissur → Kozhikode → Wayanad (7.5 h).

### 6.2 A\* — the order to visit the chosen cities (`algorithms/astar.py`)

| Element | Definition |
|---|---|
| State | `(current city or none, set of visited cities)` |
| Initial state | `(none, ∅)` |
| Actions | Go to any unvisited chosen city |
| Step cost | Hours between the two cities (from 6.1, by the chosen mode); 0 for the first city |
| Goal test | Every chosen city visited |
| Heuristic | Weight of a **minimum spanning tree** over the current city and the unvisited ones (Prim) |

*Admissible:* any way to finish is a path through those cities, and a path is a spanning tree, so it costs at least the MST. *Consistent:* adding the edge current → next to the next state's MST gives a spanning tree of this state. So A\* with a closed set returns an optimal order. Baselines on the same problem: uniform-cost search (`h = 0`), A\* with the weaker "cheapest edge out" heuristic, and greedy nearest neighbour.

### 6.3 CSP — each day's timetable (`algorithms/csp_solver.py`)

| Element | Definition |
|---|---|
| Variables | Each chosen sight, plus lunch and dinner |
| Domains | Start times in 15-minute steps: after the sight opens and the traveller can reach it, early enough to finish before it closes and before the day ends (lunch 12–14, dinner 19–21) |
| Constraints | For every pair: `start(Y) ≥ end(X) + travel(X, Y)` or `start(X) ≥ end(Y) + travel(Y, X)` |
| Search | Backtracking; **MRV** picks the variable with the fewest values left; earliest value first; **forward checking** removes clashing values from every other domain after each assignment and backtracks as soon as one is empty |

If the day cannot hold every sight, the least-wanted one is dropped and the CSP is solved again. Baseline: greedy first-fit (each sight takes the earliest slot that clashes with nothing placed; meals last, dropping the least-wanted sight if a meal has no room).

### 6.4 The other decisions

- **Interest match:** cosine similarity between the traveller's seven-theme vector and a city's profile.
- **Budget repair** (cheapest damage first): switch hotels the traveller did not choose to the cheapest; then drop the paid sight with the lowest preference per rupee (the fractional-knapsack greedy rule), keeping one sight per city; after an overrun go one tier cheaper.
- **Mode:** the traveller's choice; else the cheaper mode when the Critic sent the plan back for cost; else rail if it is no slower than the road or the drive is over the daily limit. A strike on one day sends the hops it touches to the train where one runs.

## 7. Search strategy justification

| Problem | Candidates | Chosen | Why |
|---|---|---|---|
| Travel time between two towns | BFS (ignores hours), DFS (not optimal), greedy best-first (not optimal) | **Uniform-cost search** | Positive step costs, so the first goal expansion is the shortest |
| Visiting order of N cities | Brute force (factorial), nearest neighbour (fast, not optimal), uniform-cost search (optimal, more nodes) | **A\* with the MST heuristic** | Same optimum as uniform-cost search with far fewer expansions |
| A day's timetable | Greedy first-fit (no look-ahead), integer programming (heavy for optional visits) | **CSP: backtracking + MRV + forward checking** | Windows and no-overlap are variables, domains and binary constraints; MRV places the tightest sight first; forward checking finds a dead end one step early |
| Who fixes a problem | One central planner re-running everything | **A coordinator that names one owner** | Restarting costs 1.5 to 4.7 more agent runs for the same value (measured) |

Measured on every subset of the eight cities: A\* with the MST heuristic expands 48 nodes at eight cities against 854 for uniform-cost search; nearest neighbour is optimal for all subsets up to four cities and stops being optimal from five (49 of 56 subsets), 18 of 28 at six, 2 of 8 at seven, 0 of 1 at eight.

## 8. Anticipated Q&A

| Question | Answer |
|---|---|
| Why is this multi-agent and not one program? | Six agents with different goals and different knowledge; each owns part of the state and may not touch the rest (enforced and tested); they interact only through messages, all visible in the demo's log. |
| How do agents interact? | Addressed messages on a bus. The Critic sends each problem to exactly one owner. |
| What makes the environment partially observable? | Plans are made on monthly averages. The environment tells the agents only what is real for the days, sights and journeys in the plan, and each answer is counted. |
| Is it deterministic? | The planning is. The world's surprises are seeded draws: a given seed always gives the same world and the same plan; different seeds give different worlds. |
| Is your heuristic admissible? | Yes: finishing the trip is a path through the remaining cities, a path is a spanning tree, so it costs at least the MST. Checked against brute force in the tests. |
| Why uniform-cost search for travel times? | Positive step costs, so the first time the goal is expanded its cost is the shortest; the goal test is on expansion. |
| Why a CSP for the day? | Every sight has a window and no two may overlap once travel is counted. MRV places the tightest sight first; forward checking finds a dead end early. |
| How is the replan targeted? | The Critic maps each issue to the agent that owns it and sends one REPLAN message; the agents before it are not re-run. |
| Does each part help? | Field checks: +5 to +24 points over planning once. CSP: +22 to +33 over first-fit. Targeted replanning: same value as restarting, 1.5 to 4.7 fewer runs. A\* over nearest neighbour: shows from five cities on, barely on trips this small. |
| What can it not do? | One region; averages for weather; a strike on a day that must be driven, where no train runs, cannot be avoided; each agent optimises its own part given the earlier ones, so the joint plan is not globally optimal. |
| Why no LLM here? | Non-deterministic, needs keys and quota, and hides the reasoning the course grades. The LLM version is on `main`. |
