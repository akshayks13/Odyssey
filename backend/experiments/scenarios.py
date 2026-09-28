"""The scenarios the strategies are compared on (the equivalent of the six test scenarios in Review 1).

Each scenario fixes what the environment does (the month, how rough the field is, an event the traveller
reports) and lets the seed vary the traveller: who goes, how much they spend, what they enjoy. So a
scenario is 30 different trips meeting 30 different fields, and every strategy gets exactly the same 30.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from models.schemas import Disruption, DisruptionType

INTERESTS = [
    "nature and adventure",
    "culture and food",
    "beach and relaxation",
    "nature and culture",
    "adventure and food",
    "shopping and culture",
]
PACES = ["relaxed", "moderate", "packed"]


@dataclass(frozen=True)
class Scenario:
    name: str
    what: str  # what it tests
    days: int = 5
    month: int = 1  # the trip starts on the 10th of this month, 2027
    uncertainty: str = "normal"  # the field profile (core/environment.py LEVELS)
    budget_cut: float | None = None  # if set, the traveller reports a budget cut to this share of the budget
    group: str = ""  # scenarios sharing a group are plotted as one curve (the scaling study)


SCENARIOS: list[Scenario] = [
    Scenario("calm", "A dry month, an ordinary field: the baseline. Little goes wrong, so strategies should tie.", month=1),
    Scenario("monsoon", "July: heavy rain stops boating, treks and viewpoints on many days. Does checking the field pay?", month=7),
    Scenario("closures", "Sights are shut far more often. Can the plan be repaired without changing cities?", month=1, uncertainty="closures"),
    Scenario("strikes", "Road strikes on travel days, on a 7-day trip that moves between cities.", days=7, month=1, uncertainty="strikes"),
    Scenario("budget_cut", "The traveller reports a 30% budget cut after planning. Which agent should re-plan?", month=1, budget_cut=0.7),
    Scenario("scale_3", "Trip length: 3 days", days=3, month=10, group="scaling"),
    Scenario("scale_5", "Trip length: 5 days", days=5, month=10, group="scaling"),
    Scenario("scale_8", "Trip length: 8 days", days=8, month=10, group="scaling"),
    Scenario("scale_12", "Trip length: 12 days", days=12, month=10, group="scaling"),
]
BY_NAME = {s.name: s for s in SCENARIOS}


def make_trip(scenario: Scenario, seed: int) -> tuple[str, list[Disruption]]:
    """The request text for this seed, and the events the traveller reports after planning."""
    rng = random.Random(f"{scenario.name}|{seed}")
    people = rng.choice([2, 3, 4])
    per_person_per_day = rng.randint(2600, 4200)
    budget = int(round(people * scenario.days * per_person_per_day, -3))
    text = (
        f"{scenario.days} days in Kerala for {people} from 2027-{scenario.month:02d}-10, ₹{budget:,}, "
        f"{rng.choice(INTERESTS)} at a {rng.choice(PACES)} pace"
    )
    events = []
    if scenario.budget_cut:
        events.append(Disruption(type=DisruptionType.BUDGET_CUT, target="trip", description="budget cut", new_budget_inr=round(budget * scenario.budget_cut, -3)))
    return text, events
