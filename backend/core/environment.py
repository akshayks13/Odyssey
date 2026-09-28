"""The world the agents cannot fully see.

`FieldWorld` is the ground truth: which days really have heavy rain, which sights are really closed on a
given day, which road transport is really on strike. It is *stochastic but reproducible*: every fact is a
seeded draw (a hash of seed + kind + key), so the same seed always gives the same world, and every strategy
in an experiment faces exactly the same one.

The agents plan on averages (a month's typical weather). `Environment` is the sensor: when the Architect
hands over a schedule, it looks up only the days, sights and journeys in that schedule and reports what is
real. Facts it was never asked about stay unknown (partial observability), and each fact asked about is
counted as a field check.
"""
from __future__ import annotations

import zlib
from datetime import date

from agents.base import Post, Result
from core.messages import Message, MsgType
from models.schemas import Disruption, DisruptionType, ObservedFacts, TransportMode
from tools import world

# Chances used when the environment is switched on ("normal"). Heavy rain is the share of "rain likely" days on
# which boating, treks and viewpoints are actually stopped; closures are per sight per day; strikes per city per day.
HEAVY_RAIN_SHARE = 0.5
SIGHT_CLOSURE_PROB = 0.05
STRIKE_PROB = 0.04

# How much to scale each kind of surprise. "off" is a world where nothing goes wrong; "closures" and "strikes" stress one
# kind at a time (used by the experiments).
LEVELS: dict[str, dict[str, float] | None] = {
    "off": None,
    "normal": {"rain": 1.0, "closure": 1.0, "strike": 1.0},
    "high": {"rain": 1.6, "closure": 3.0, "strike": 3.0},
    "closures": {"rain": 1.0, "closure": 6.0, "strike": 1.0},
    "strikes": {"rain": 1.0, "closure": 1.0, "strike": 8.0},
}


class FieldWorld:
    """Ground truth. Deterministic given (seed, level)."""

    def __init__(self, seed: int = 0, level: str = "normal") -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown uncertainty level {level!r}; use one of {sorted(LEVELS)}")
        self.seed = seed
        self.level = level
        self.rates = LEVELS[level]

    @property
    def enabled(self) -> bool:
        return self.rates is not None

    def _u(self, kind: str, key: str) -> float:
        """A repeatable uniform draw in [0, 1) for one fact."""
        return zlib.crc32(f"{self.seed}|{kind}|{key}".encode()) / 2**32

    def heavy_rain(self, region: str, city: str, day: str) -> bool:
        if not self.enabled:
            return False
        chance = world.weather_on(region, city, date.fromisoformat(day))["rain_chance"] / 100
        return self._u("rain", f"{city}|{day}") < min(1.0, chance * HEAVY_RAIN_SHARE * self.rates["rain"])

    def sight_closed(self, sight_id: str, day: str) -> bool:
        return self.enabled and self._u("closed", f"{sight_id}|{day}") < min(1.0, SIGHT_CLOSURE_PROB * self.rates["closure"])

    def strike(self, city: str, day: str) -> bool:
        return self.enabled and self._u("strike", f"{city}|{day}") < min(1.0, STRIKE_PROB * self.rates["strike"])


class Environment:
    """The sensor. Not one of the six agents: it has no goal, it only answers what it is asked."""

    name = "environment"
    label = "Field environment"
    reads = ("trip_spec", "draft_itinerary", "observed", "disruptions")
    writes = ("observed", "disruptions")

    def __init__(self, truth: FieldWorld) -> None:
        self.truth = truth

    def handle(self, msg: Message, view: dict) -> Result:
        itinerary, spec = view["draft_itinerary"], view["trip_spec"]
        observed: ObservedFacts = view["observed"].model_copy(deep=True)
        disruptions: list[Disruption] = list(view["disruptions"])
        region = spec.destination_region
        found: list[str] = []
        asked_before = observed.checks

        def ask(fact: str) -> bool:
            if fact in observed.checked:
                return False
            observed.checked.append(fact)
            observed.checks += 1
            return True

        for day in itinerary.days:
            if not day.date:
                continue
            city, when = day.destination, day.date
            if ask(f"rain:{city}|{when}"):
                if self.truth.heavy_rain(region, city, when):
                    observed.heavy_rain.append(f"{city}|{when}")
                    found.append(f"heavy rain in {city} on day {day.day_number}")
                else:
                    observed.dry.append(f"{city}|{when}")
            for item in day.items:
                if item.kind == "activity" and ask(f"closed:{item.activity_id}|{when}") and self.truth.sight_closed(item.activity_id, when):
                    observed.closed.append(f"{item.activity_id}|{when}")
                    found.append(f"{item.activity_name} closed on day {day.day_number}")
            leg = day.travel_leg
            if leg and leg.mode == TransportMode.ROAD:
                for end in (leg.origin, leg.destination):
                    if ask(f"strike:{end}|{when}") and self.truth.strike(end, when):
                        found.append(f"road strike at {end} on day {day.day_number}")
                        if not any(d.type == DisruptionType.TRANSPORT and d.target == end and d.date == when for d in disruptions):
                            disruptions.append(Disruption(type=DisruptionType.TRANSPORT, target=end, description=f"Road strike on {when}", day=day.day_number, date=when))

        asked = observed.checks - asked_before
        summary = ("; ".join(found) if found else "everything checked is as expected")
        message = f"Field check: asked about {asked} new fact(s): {summary}."
        return Result(
            updates={"observed": observed, "disruptions": disruptions},
            posts=[Post(MsgType.FIELD_REPORT, "critic_replanner", f"{asked} checked: {summary}", {"found": found})],
            message=message,
            tools=["heavy_rain?", "sight_closed?", "strike?"],
            algorithms=["seeded field lookups (only for what the plan uses)"],
            note=f"{observed.checks} facts checked so far",
        )
