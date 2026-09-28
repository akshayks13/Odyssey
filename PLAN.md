# Odyssey — design (deterministic multi-agent version)

Branch `simple-deterministic`. Six cooperating agents, hand-written search and constraint solving, a seeded
environment that hides part of the truth. No language model, no external API, no framework doing the thinking.
How to run it is in [`README.md`](README.md). The LLM version is on `main`.

## 1. Problem

A traveller types what they want ("5 days in Kerala with 3 friends, ₹60,000, nature and adventure, relaxed pace").
The team must produce a day-by-day itinerary: cities they will like, in a sensible order, within budget, with every
sight inside its opening hours. But the plan is made on **averages** (a month's typical weather), and the world is not
average: on some days it rains hard, a sight is shut, a road strike hits a travel day, the traveller cuts the budget.
The agents only find out what is real by **checking**, and when they do, only the agent that owns the problem should
have to redo its work.

**Objectives.**
1. Model the trip as a partially observable, stochastic, dynamic environment with a known, seeded model of surprises.
2. Build agents with distinct roles that share no code paths: they only read and write the parts of the state they own,
   and they talk through an explicit message protocol.
3. Use informed search (A\*) for the visiting order, uniform-cost search for travel times, and a constraint satisfaction
   solver (backtracking, MRV, forward checking) for each day's timetable.
4. Show, by controlled experiments over many seeds, that each design choice earns its place, and say where it does not.

**Scope.** One region (Kerala), modelled by hand in `backend/data/kerala.json`. Prices, times and weather averages are
approximate and illustrative; hotel names are placeholders.

## 2. System overview

```
                      ┌────────────────────────── MessageBus ──────────────────────────┐
traveller ─REQUEST──► trip_analyst ─SPEC_READY──► destination_agent ─CITIES_READY──► mobility_agent
                                                                                        │ ROUTE_READY
traveller ◄─ACCEPT / BEST_EFFORT─ critic_replanner ◄─FIELD_REPORT─ environment ◄─SCHEDULE_READY─ itinerary_architect ◄─BUDGET_READY─ budget_agent
                     │  ▲                                                                                 ▲
                     │  └─DISRUPTION─ traveller                                     REPLAN (to the ONE owner) ┘  (any of the four)
                     └─────────────────────────────────────────────────────────────────────────────────┘
traveller ─EDIT──► edit_router ─RERUN──► the earliest agent whose inputs changed   (or ANSWER ► traveller)
```

- **The agents never call each other.** Every arrow is a message on a bus (`backend/core/messages.py`): `REQUEST`,
  `SPEC_READY`, `NEED_INFO`, `CITIES_READY`, `ROUTE_READY`, `BUDGET_READY`, `SCHEDULE_READY`, `FIELD_REPORT`, `REPLAN`,
  `RERUN`, `ACCEPT`, `BEST_EFFORT`, `DISRUPTION`, `EDIT`, `ANSWER`. The demo shows the whole conversation live.
- **One shared blackboard, with ownership.** Each agent declares the keys it `reads` and `writes`
  (`backend/core/blackboard.py`). It is handed a view containing only its reads (asking for anything else raises), and
  its updates are refused if they touch a key it does not own. The tests run whole sessions under this enforcement.
- **The runtime makes no decisions.** `Team` (`backend/core/team.py`) takes a message off the bus, hands it to the agent
  it is addressed to, stores the agent's updates, and puts the agent's replies back on the bus. Who hears about a
  problem is decided by the Critic, not by a router or a framework.
- **The environment is a sensor, not an agent.** It has no goal. It answers only what it is asked about (section 4).

## 3. PEAS, per agent (Review 1)

All agents share one system performance measure, so their individual goals pull the same way:

`S = w_p·P + w_q·Q + w_r·R + w_b·B − w_t·T − w_c·C`

P = interest match of the scheduled sights (cosine), Q = their rating, R = route efficiency (1 − share of time spent
travelling), B = budget efficiency, T = travel burden, C = constraint violations. The weights depend on the traveller's
archetype (adventure / relaxed / budget / balanced), inferred from their interests and budget.

The tables below are generated from the `peas`, `reads` and `writes` declared on each agent class
(`GET /api/agents` returns the same data), and a test checks that this file contains them.

### Trip Analyst

*Parser: turns the traveller's sentence into a trip spec.* Architecture: **simple reflex (condition-action rules on the text)**.

| PEAS | |
|---|---|
| Performance | Every stated fact read correctly (days, people, budget, dates, interests, pace, travel mode); every unstated fact reported as an assumption; never a guessed place |
| Environment | The traveller's sentence and the list of regions the dataset covers |
| Actuators | Write the trip spec; ask the traveller a question (NEED_INFO); send SPEC_READY |
| Sensors | The request text; the region and city index |

Reads `raw_input`, `edit_locks`. Writes `trip_spec`, `edit_locks`, `assistant_reply`.

### Destination Discovery

*Explorer: ranks the region's cities for this traveller and picks the stops.* Architecture: **goal-based (finds a set of stops that satisfies the interest and distance goals)**.

| PEAS | |
|---|---|
| Performance | High interest match; every stop within a day's travel of another; no closed, avoided or storm-hit city; enough sights for the days spent there |
| Environment | The region's cities, their interest profiles, the road network, monthly weather, and the traveller's standing choices |
| Actuators | Write the ranked cities, the chosen stops and their sights; send CITIES_READY |
| Sensors | The trip spec, the edit locks, reported disruptions, the Critic's REPLAN directive |

Reads `trip_spec`, `edit_locks`, `disruptions`, `selected_destinations`. Writes `candidate_destinations`, `selected_destinations`, `candidate_activities`, `excluded_activity_ids`.

### Mobility & Routing

*Mover: chooses road or rail for each hop and the order in which to visit the cities.* Architecture: **goal-based (searches for the fastest complete visiting order)**.

| PEAS | |
|---|---|
| Performance | Minimum total travel time between cities; no hop longer than the daily limit unless asked for; the mode the traveller asked for; fares that leave room for the rest of the budget |
| Environment | The road and rail networks (with junction towns), fares, transport strikes, and the traveller's travel-mode choice |
| Actuators | Write the route (order, legs, mode, cost); send ROUTE_READY |
| Sensors | The chosen cities, the trip spec, the edit locks, reported strikes, the Critic's REPLAN directive |

Reads `trip_spec`, `edit_locks`, `selected_destinations`, `disruptions`. Writes `route`.

### Budget Optimization

*Money: prices hotels, food, sights and transport against the ceiling and repairs an overrun.* Architecture: **utility-based (trades comfort for money by the cheapest damage per rupee saved)**.

| PEAS | |
|---|---|
| Performance | Total at or under the ceiling; the fewest and least valuable sights dropped; the traveller's hotel choices kept |
| Environment | Hotel prices per tier, food and local-transport prices, fares from Mobility, the ceiling and any reported budget cut |
| Actuators | Write the budget breakdown, the hotels, the stay plan and the sights dropped; send BUDGET_READY |
| Sensors | The route and its fares, the candidate sights, the trip spec, the edit locks, reported budget cuts, the previous breakdown |

Reads `trip_spec`, `selected_destinations`, `route`, `candidate_activities`, `edit_locks`, `disruptions`, `budget_breakdown`. Writes `budget_breakdown`, `accommodation_options`, `excluded_activity_ids`, `stay_plan`.

### Itinerary Architect

*Scheduler: packs sights and meals into each day and says where each night is spent.* Architecture: **goal-based (finds a consistent timetable: a constraint satisfaction problem per day)**.

| PEAS | |
|---|---|
| Performance | Every sight inside its opening hours and the day's window; no overlaps once travel time is counted; meals in their windows; the traveller's free, light and pinned days kept; high score |
| Environment | Opening hours, distances inside a city, the stay plan, the weather (expected and reported), sights reported closed |
| Actuators | Write the day-by-day schedule and the reconciled cost of what it scheduled; send SCHEDULE_READY |
| Sensors | The route, the sights and their hours, the budget, the stay plan, the edit locks, the field reports |

Reads `trip_spec`, `route`, `candidate_activities`, `budget_breakdown`, `selected_destinations`, `excluded_activity_ids`, `edit_locks`, `stay_plan`, `observed`. Writes `draft_itinerary`, `optimization_score`, `budget_breakdown`.

### Critic & Replanner

*Coordinator: validates the draft against the rules and the field, and sends each problem to the one agent that owns it.* Architecture: **utility-based (picks the most severe issue and the cheapest agent to fix it)**.

| PEAS | |
|---|---|
| Performance | Every real problem caught; each sent to exactly one owner; the fewest replan loops; a valid plan accepted; the best plan so far returned when no retry can help |
| Environment | The draft plan, the budget, reported events, the field report, and the other agents' declared ownership |
| Actuators | Write the validation report; send REPLAN to one agent, or ACCEPT / BEST_EFFORT to the traveller; fold handled events into standing choices |
| Sensors | Everything on the blackboard (read only), incoming SCHEDULE_READY, FIELD_REPORT and DISRUPTION messages |

Reads `draft_itinerary`, `budget_breakdown`, `disruptions`, `iteration_count`, `trip_spec`, `route`, `edit_locks`, `selected_destinations`, `candidate_activities`, `excluded_activity_ids`, `replan_directives`, `edit_directive`, `conflicts`, `observed`. Writes `validation_report`, `replan_directives`, `iteration_count`, `conflicts`, `final_itinerary`, `disruptions`, `edit_locks`, `trip_spec`.

### Edit Router

*Front door for changes: reads a typed change and starts the pipeline again at the earliest agent it affects.* Architecture: **simple reflex (condition-action rules on the text)**.

| PEAS | |
|---|---|
| Performance | Every change understood and stored as a standing choice; the pipeline re-entered no later than the earliest agent whose inputs changed; questions answered without touching the plan |
| Environment | The current plan, the traveller's standing choices, and one typed sentence |
| Actuators | Update the trip spec and the edit locks; send RERUN to one agent, or ANSWER to the traveller |
| Sensors | The plan (itinerary, budget, route, cities, sights) and the EDIT message |

Reads `trip_spec`, `edit_locks`, `selected_destinations`, `candidate_activities`, `disruptions`, `final_itinerary`, `draft_itinerary`, `budget_breakdown`, `route`. Writes `trip_spec`, `edit_locks`, `edit_directive`, `disruptions`, `iteration_count`, `replan_directives`, `final_itinerary`, `assistant_reply`.

## 4. Environment analysis (Review 1)

| Property | Value | Why | What it forces in the design |
|---|---|---|---|
| Observable | **Partially** | The dataset (roads, hours, prices, averages) is known. What is real on a given day is not: heavy rain, a closed sight, a strike. The environment answers only for the days, sights and journeys in the plan it is shown | Plan on averages, then check; keep what has been learned (`observed`); count the checks |
| Deterministic | **Stochastic, but reproducible** | Surprises are seeded draws (a hash of seed, kind and key), so the same seed always gives the same world | A plan is repaired, not assumed final; every strategy in an experiment meets the same field, so results are paired |
| Sequential | **Yes** | Cities change the route, the route's fares change the budget, the budget's cuts change the schedule | An order of work; a repair may need to re-run what follows it |
| Static / dynamic | **Dynamic** | Events arrive after the plan exists: reported closures, storms, strikes, budget cuts, typed edits | A Critic that reacts, and targeted replanning so a small event costs a small amount of work |
| Discrete | **Yes** | Towns are graph nodes, time is 15-minute slots, tiers are three values | Search over finite spaces is well defined |
| Agents | **Multi-agent, cooperative** | Six agents with different goals (interest, travel time, money, feasibility) and one shared score | Ownership of state, a message protocol, and a coordinator for conflicts |
| Rules | **Known rules, unknown state** | The model of surprises is known, which draw comes up is not | Expected values guide the plan; observation corrects it |

**The field** (`backend/core/environment.py`). Heavy rain: each day, with probability
`monthly rain chance × 0.5 × level`; it stops boating, treks and viewpoints (the "outdoor" sights). A sight closed:
5% per sight per day. A road strike: 4% per city per day, felt on a day that drives in or out of it. Levels: `off`,
`normal`, `high`, and `closures` / `strikes` (one kind of surprise turned up, for the experiments). Measured
frequencies are checked in the tests (July rain about 42%, January about 5%).

**Sensing has a cost.** When the Architect hands over a schedule, the environment looks up only the days, sights and
journeys in it. Each fact is asked once and remembered; the number of facts asked is reported as *field checks*.

## 5. Agent analysis (Review 1)

A simple reflex agent (condition-action rules, no memory) is enough to *parse text*: the Trip Analyst and the Edit
Router. It is not enough for the rest, because the right action depends on state that is not in the current percept: a
sight's opening hours against the day already built, the fares already committed, what the field has already reported.

- **Goal-based**: Destination (a set of stops meeting the interest and distance goals), Mobility (the fastest complete
  order), Architect (a consistent timetable). Each searches for a state that satisfies a goal.
- **Utility-based**: Budget (the least damage per rupee saved), Critic (the most severe issue, the cheapest owner). Each
  compares outcomes numerically rather than testing a goal.

## 6. Algorithms (Review 1: modelling and search strategy)

### 6.1 Uniform-cost search — travel time between two towns (`algorithms/search.py`)

- **State** a town. **Actions** take one road (or rail) link. **Step cost** hours, always positive. **Goal** the
  destination. The frontier is ordered by path cost `g(n)`, and the goal is tested when a node is *expanded*, not when
  it is generated.
- Complete and optimal because every step cost is positive. The networks are sparse and include junction towns, so
  Kochi → Wayanad is found as Kochi → Thrissur → Kozhikode → Wayanad (7.5 h).

### 6.2 A\* — the order to visit the chosen cities (`algorithms/astar.py`)

- **State** `(current city or None, frozenset of visited cities)`; **initial** `(None, ∅)`. **Actions** go to any
  unvisited city. **Step cost** hours between the two cities (from 6.1, by the chosen mode), 0 for the first city.
  **Goal** all cities visited. An open travelling-salesman path.
- **Heuristic**: weight of a **minimum spanning tree** over the current city and the unvisited ones (Prim).
  *Admissible*: any way to finish is a path through those cities, and a path is a spanning tree, so it costs at least
  the MST. *Consistent*: adding the edge current→next to the next state's MST gives a spanning tree of this state. So
  A\* with a closed set returns an optimal order.
- **Baselines** on the same problem: uniform-cost search (`h = 0`), A\* with the weaker "cheapest edge out" heuristic,
  and greedy nearest neighbour. (`python -m experiments.search_comparison`, results below.)

### 6.3 CSP — each day's timetable (`algorithms/csp_solver.py`)

- **Variables** each chosen sight, plus lunch and dinner. **Domains** start times in 15-minute steps, after the sight
  opens and after the traveller can reach it from the hotel, early enough to finish before it closes and before the day
  ends (lunch 12–14, dinner 19–21). **Constraints** for every pair: `start(Y) ≥ end(X) + travel(X, Y)` or
  `start(X) ≥ end(Y) + travel(Y, X)`.
- **Search** backtracking; **MRV** picks the variable with the fewest values left; values are tried earliest first (a
  compact day); **forward checking** removes clashing values from every other domain after each assignment and
  backtracks as soon as one is empty. If the day cannot hold every sight, the least-wanted one is dropped and the CSP is
  solved again.
- **Baseline**: greedy first-fit. Sights in preference order each take the earliest slot that clashes with nothing
  placed; meals are then placed, dropping the least-wanted sight if a meal finds no room. It honours the same
  constraints but has no look-ahead and never rearranges.

### 6.4 The other decisions

- **Interest match**: cosine similarity between the traveller's seven-theme vector and a city's profile.
- **Budget repair**, cheapest damage first: switch hotels the traveller did not choose to the cheapest, biggest saving
  first; then drop the paid sight with the lowest preference per rupee (the fractional-knapsack greedy rule), keeping at
  least one sight per city. After an overrun, go one tier cheaper.
- **Mode**: the traveller's choice; else the cheaper mode when the Critic sent the plan back for cost; else rail if it is
  no slower than the road or the drive is over the daily limit. A strike on one day sends the hops it touches to the
  train where one runs.

## 7. Coordination and conflict resolution (Review 2)

**Who fixes what** (the Critic, one owner per loop, at most five loops, a reported event first, then by severity):

| Issue | Sent to | What that agent does differently |
|---|---|---|
| CLOSURE / WEATHER reported at a city | Destination | keeps the cities that are fine, replaces that one |
| TRAVEL_OVERLOAD (a hop over the daily limit) | Destination | avoids the far city (unless the traveller named both cities: then only noted) |
| LEISURE_DAY (nothing left to see) | Destination | takes one more city |
| TRANSPORT (a strike) | Mobility | slower road for a reported strike; train where one runs for a one-day strike |
| BUDGET, fares ≥ 40% of the budget | Mobility | cheapest mode per hop |
| BUDGET | Budget | one tier cheaper |
| BUDGET_CUT | Budget | re-prices against the new ceiling |
| SCHEDULE_CONFLICT | Architect | re-solves the day |
| HEAVY_RAIN reported on a day with outdoor sights | Architect | drops outdoor sights from that day (they stay available for later days) |
| SIGHT_CLOSED reported for a scheduled sight | Architect | drops that sight for that day |

An issue still present after its agent already retried is not sent again (a field issue is, when it is a new fact). A
low-severity issue is a note, not a fault. After the loop limit the best plan so far is returned with what is left.

**Conflicts between agents** are recorded and resolved by the Critic, not by overwriting:

| Conflict | Resolution |
|---|---|
| Destination's favourite city vs Budget's overrun | Cheapest-damage repair first (hotel tier, or travel mode), before dropping the city |
| Budget priced the sights before they were scheduled; the Architect scheduled different ones | The Architect reconciles the activity line to what it really scheduled (the one key two agents may write, declared as such) |
| A closure vs a traveller's pinned city | The closure wins; the pin is removed and the closed city is excluded for good |
| Two edits that disagree | The later one replaces the earlier for the same lock |
| Field report contradicts the average | The report wins for that day; the average stays for days not checked |

**Standing choices.** `EditLocks` (pinned and excluded cities, mode, hotels, skipped and pinned sights, free and light
days, pace, start time) are read by every agent, so a later replan never undoes an edit. A handled closure or budget cut
is folded into them too.

**Strategies.** Each switches off one design choice; the experiments compare them:

| Strategy | What it changes |
|---|---|
| `full` | The system: A\* order, CSP timetable, targeted replanning, field checks |
| `greedy_order` | Nearest-neighbour order instead of A\* |
| `greedy_schedule` | First-fit timetable instead of the CSP |
| `restart` | Any issue re-runs every agent from Destination (no diagnosis) |
| `static` | Plans once; no field checks, no replanning (independent agents, no Critic loop) |

## 8. Tools and packages (Review 2)

| Tool | Why |
|---|---|
| Python 3.11+ (tested on 3.13) | Course language |
| `heapq` (standard library) | The priority queue for UCS and A\*. The searches and the CSP are hand-written because they are the graded part |
| Pydantic | Validated schemas for every piece of state |
| FastAPI + server-sent events | Streams every message, step and result to the UI as the agents work |
| Next.js, Tailwind, MapLibre, Recharts | The web UI: agent timeline, message log, day-by-day plan, budget chart, route map, strategy comparison |
| matplotlib | The experiment charts (`requirements-dev.txt`; the tables need nothing) |
| pytest | 126 tests, offline, about two seconds |

**Considered and not used.** An LLM (non-deterministic, needs keys and quota, and hides the reasoning the course
grades). LangGraph or another agent framework (the message bus and the ownership rules *are* the multi-agent design,
and a framework would hide them). OR-Tools and NetworkX (they would do the search and the CSP we are meant to write).
Live travel APIs (quota, and results change between runs). Mesa (built for grid simulations; our world is a graph).

## 9. Experiments (Review 2)

`python -m experiments.run_experiments` runs 9 scenarios × 5 strategies × 30 seeds = 1,350 runs in a few seconds.
For every (scenario, seed) all five strategies plan the same trip, meet the same field and hear the same reported
events. `value_ratio` is the share of the trip's value that survives contact with the field: the value of the sights that
really work ÷ the value of the plan the same system makes in a world where nothing goes wrong. A sight is lost if it is
outdoors on a heavy-rain day, is shut that day, is in a city reported closed or hit by a storm, or falls on a
strike day; the value left is scaled down by the share the plan overshoots the budget the traveller now has.

| Scenario | Tests |
|---|---|
| `calm` | A dry month, an ordinary field: the baseline |
| `monsoon` | July: heavy rain on many days. Does checking the field pay? |
| `closures` | Sights shut far more often. Can the plan be repaired without changing cities? |
| `strikes` | A 7-day trip that moves between cities, with frequent road strikes |
| `budget_cut` | The traveller reports a 30% cut after planning. Which agent should re-plan? |
| `scale_3/5/8/12` | Trip length, 3 to 12 days |

Seeds vary the traveller (who goes, how much they spend, what they enjoy, the pace), so each scenario is 30 different
trips. Charts and CSVs are in [`backend/results/`](backend/results/).

### Results (mean % of the no-surprise trip's value delivered, 30 seeds)

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

Paired on the same seed (full system minus the baseline, ± the 95% interval; \* = the interval excludes zero):

| Scenario | vs plan once | vs greedy timetable | vs greedy order | vs restart (value) | agent runs saved vs restart |
|---|---|---|---|---|---|
| calm | +5.4 ± 3.0 \* | +31.2 ± 5.0 \* | +1.5 ± 1.8 | 0.0 | +1.5 ± 0.6 \* |
| monsoon | +6.1 ± 2.8 \* | +28.1 ± 5.1 \* | +2.0 ± 2.2 | 0.0 | +2.4 ± 0.6 \* |
| closures | +19.6 ± 5.3 \* | +26.3 ± 4.3 \* | +2.2 ± 2.5 | 0.0 | +4.7 ± 0.8 \* |
| strikes | +11.5 ± 5.1 \* | +23.5 ± 4.2 \* | +2.8 ± 5.7 | +3.3 ± 3.4 | +2.4 ± 0.9 \* |
| budget_cut | +23.7 ± 3.8 \* | +27.7 ± 3.2 \* | +0.1 ± 1.2 | −0.7 ± 1.1 | +3.7 ± 0.7 \* |
| scale_3 | +6.7 ± 4.4 \* | +33.1 ± 5.9 \* | −0.7 ± 4.8 | 0.0 | +2.0 ± 0.5 \* |
| scale_5 | +5.8 ± 3.0 \* | +30.0 ± 4.4 \* | +1.8 ± 1.5 \* | 0.0 | +1.9 ± 0.5 \* |
| scale_8 | +17.7 ± 7.2 \* | +28.2 ± 3.9 \* | 0.0 | 0.0 | +3.3 ± 1.2 \* |
| scale_12 | +19.6 ± 6.1 \* | +22.0 ± 3.3 \* | −2.5 ± 3.4 | +9.2 ± 5.7 \* | +3.6 ± 1.1 \* |

### What the results say

- **Checking the field and repairing pays, everywhere.** The full system beats planning once in all nine scenarios (+5 to
  +24 points, the interval excludes zero every time). The gain is largest where the surprise is large: closures,
  a budget cut that a plan-once team ignores, long trips with many days to go wrong.
- **The CSP matters a lot.** Replacing it with a first-fit timetable costs 22 to 33 points in every scenario, even in a
  calm world. The baseline is deliberately simple (no look-ahead); a smarter greedy heuristic would narrow the gap, but
  first-fit loses sights whose opening windows collide, which is exactly what backtracking with MRV avoids.
- **Targeted replanning gives the same quality for less work.** Restarting from Destination reaches exactly the same value
  in six of the nine scenarios and is within noise in two more (the full system 3 points ahead in `strikes`, restart 0.7
  ahead in `budget_cut`), but it uses 1.5 to 4.7 more agent runs per trip, significantly so in every scenario. On the
  12-day trips restart is also 9 points *worse*: re-choosing the cities discards the stable parts of a long plan.
- **A\* barely shows up end to end, and we say so.** Against nearest-neighbour ordering the full system is ahead in six
  scenarios, level in one and behind in two, and the interval excludes zero in only one. The trips visit 2 to 6 places,
  and on this small road network nearest-neighbour is optimal for most small sets. A\*'s advantage is real, and it shows where the order is
  hard: at 5+ cities nearest-neighbour stops being optimal (table below), and A\*'s 48 node expansions at 8 cities
  against 854 for uniform-cost search.
- **Some losses cannot be fixed.** In `strikes` the full system still delivers only 78%: a strike on a day that must be
  driven, in a place with no railway, cannot be avoided (the Critic reports it and returns the best plan).

### A\* against uniform-cost search and greedy (every subset of the eight cities)

| Cities | Subsets | A\* (MST) nodes | A\* (min-edge) nodes | UCS nodes | A\*/UCS optimal | Greedy optimal | Greedy extra hours |
|---|---|---|---|---|---|---|---|
| 4 | 70 | 9.8 | 16.3 | 26.7 | yes | 70/70 | 0.00 |
| 5 | 56 | 15.9 | 44.4 | 67.8 | yes | 49/56 | 0.14 |
| 6 | 28 | 25.1 | 115.5 | 164.6 | yes | 18/28 | 0.53 |
| 7 | 8 | 36.9 | 282.6 | 383.6 | yes | 2/8 | 1.15 |
| 8 | 1 | 48.0 | 649.0 | 854.0 | yes | 0/1 | 2.00 |

All three A\*/UCS variants always find the optimum (also checked against brute force in the tests); the MST heuristic
expands about 18× fewer nodes than UCS at eight cities.

## 10. Testing scenarios (Review 2)

126 tests, offline, about two seconds. The ones that carry the argument:

| # | What is checked | Test |
|---|---|---|
| 1 | Two towns with no direct road: UCS finds the path through junctions | `test_ucs_finds_the_path_through_junction_towns` |
| 2 | A\* order equals the brute-force optimum on every 2–5 city subset | `test_astar_is_optimal_on_every_subset` |
| 3 | The MST heuristic never overestimates, on every partial state | `test_mst_heuristic_never_overestimates` |
| 4 | The CSP respects opening hours, meals and travel; drops the least-wanted sight when a day is too full | `test_csp_*` |
| 5 | First-fit loses a sight the CSP keeps (no look-ahead) | `test_first_fit_has_no_lookahead_so_it_loses_a_sight_the_csp_keeps` |
| 6 | A plan is a chain of addressed messages | `test_a_plan_is_a_chain_of_addressed_messages` |
| 7 | An agent may not write what it does not own, and sees only what it declared | `test_an_agent_may_not_write_…`, `test_an_agent_sees_only_…` |
| 8 | Every strategy survives whole sessions (plans, events, all edits) under strict read/write enforcement | `test_agents_stay_inside_their_declared_reads_and_writes_…` |
| 9 | The field is reproducible, seed-dependent, and its frequencies match the rates | `test_the_field_*` |
| 10 | The team learns only about what the plan uses | `test_the_team_only_learns_about_what_the_plan_uses` |
| 11 | Heavy rain is found and only the Architect replans; what is scheduled then survives the real weather | `test_heavy_rain_is_found_and_only_the_architect_replans` |
| 12 | Plan-once loses sights the full system saves; restarting costs more runs than a targeted replan | `test_the_static_strategy_…`, `test_restarting_costs_more_…` |
| 13 | A closure goes Critic → Destination only; the other cities stay | `test_closing_a_city_replans_from_destination_only` |
| 14 | A budget cut goes to Budget; a strike to Mobility | `test_budget_cut_reprices_from_budget`, `test_strike_replans_the_route` |
| 15 | Edits re-run from the earliest affected agent; a question leaves the plan alone | `test_edits_rerun_only_from_…`, `test_a_question_leaves_the_plan_alone` |
| 16 | The same request meets the same field and gives the same plan | `test_same_request_*` |
| 17 | The experiment harness: every strategy on every seed, paired, reproducible | `tests/test_experiments.py` |
| 18 | This document contains every agent's PEAS | `test_this_design_document_lists_every_agents_peas` |

## 11. Viva quick answers

| Question | Answer / show |
|---|---|
| Why is this multi-agent and not one program? | Six agents with different goals and different knowledge; each owns part of the state and may not touch the rest (enforced and tested); they only interact through messages, all visible in the demo's log |
| How do agents interact? | Addressed messages on a bus; the Critic sends each problem to exactly one owner |
| What makes the environment partially observable? | Plans are made on monthly averages; the environment tells the agents only what is real for the days, sights and journeys in the plan, and each answer is counted |
| Is it deterministic? | The planning is. The world's surprises are seeded draws, so a given seed always gives the same world and the same plan; different seeds give different worlds |
| Is the A\* heuristic admissible? | Yes: finishing the trip is a path through the remaining cities, a path is a spanning tree, so it costs at least the MST. Checked against brute force in the tests |
| Why UCS for travel times? | Positive step costs, so the first time the goal is expanded its cost is the shortest; goal test on expansion |
| Why a CSP for the day? | Every sight has a window and no two may overlap once travel is counted: variables, domains, binary constraints. MRV places the tightest sight first; forward checking finds a dead end one step early |
| How is the replan targeted? | The Critic maps each issue type to the agent that owns it and sends one REPLAN message; the agents before it are not re-run. Restarting instead costs 1.5–4.7 more agent runs per trip |
| Does each part help? | Field checks: +5 to +24 points. CSP: +22 to +33. Targeted replanning: same value, fewer runs. A\*: shows at 5+ cities, barely on trips this small (honest result) |
| What can it not do? | One region; averages for weather; a strike on a road-only day cannot be avoided; each agent optimises its own part given the earlier ones, so the joint plan is not globally optimal |
| Why no LLM? | Non-deterministic, needs keys and quota, and hides the reasoning; the course grades the algorithms. The LLM version is on `main` |
