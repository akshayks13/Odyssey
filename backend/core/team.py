"""The team: six agents, one environment, one blackboard and one message bus.

`Team` is the runtime, not a decision maker. It takes a message off the bus, hands it to the agent it is
addressed to (with a view of the blackboard limited to what that agent reads), stores the agent's updates
(refusing anything the agent does not own), and puts the messages the agent sent back on the bus. It stops
when the bus is empty. Every decision, including who hears about a problem, is made by an agent.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Iterator

from agents.budget_agent import BudgetAgent
from agents.critic_replanner import CriticAgent
from agents.destination_agent import DestinationAgent
from agents.edit_router import EditRouter
from agents.itinerary_architect import ItineraryArchitect
from agents.mobility_agent import MobilityAgent
from agents.trip_analyst import TripAnalyst
from core.blackboard import Blackboard, initial_state
from core.environment import Environment, FieldWorld
from core.messages import TRAVELLER, MessageBus, MsgType
from core.snapshot import serialize_final_state
from core.strategies import STRATEGIES, Strategy
from models.schemas import Disruption


class Team:
    def __init__(self, strategy: str | Strategy = "full", seed: int = 0, uncertainty: str = "normal") -> None:
        base = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
        self.truth = FieldWorld(seed, uncertainty)
        # Field checks only happen when the strategy asks for them and the world can surprise anyone.
        self.strategy = dataclasses.replace(base, observe=base.observe and self.truth.enabled)
        self.seed = seed
        self.uncertainty = uncertainty
        self.bb = Blackboard()
        self.bus = MessageBus()
        agents = (TripAnalyst, DestinationAgent, MobilityAgent, BudgetAgent, ItineraryArchitect, CriticAgent, EditRouter)
        self.agents = {a.name: a for a in (cls(self.strategy) for cls in agents)}
        self.environment = Environment(self.truth)
        self.runs: Counter[str] = Counter()
        self.request = ""
        self.reported: list[Disruption] = []  # what the traveller has reported, so a comparison can replay it

    # -- what the traveller can do -------------------------------------------------

    def plan(self, text: str) -> Iterator[dict]:
        """Start a plan from a sentence."""
        self.request = text
        self.reported = []
        self.bb.write(**initial_state(text))
        self.bus.post(MsgType.REQUEST, TRAVELLER, "trip_analyst", text[:80], text=text)
        return self._drain()

    def disrupt(self, disruption: Disruption) -> Iterator[dict]:
        """Report an event (a closure, a storm, a strike, a budget cut). It goes to the Critic."""
        self.reported.append(disruption)
        self.bb.write(disruptions=[*self.bb["disruptions"], disruption], iteration_count=0, edit_directive=None, assistant_reply=None, replan_directives=[])
        self.bus.post(MsgType.DISRUPTION, TRAVELLER, "critic_replanner", f"{disruption.type.value} at {disruption.target}: {disruption.description}"[:90])
        return self._drain()

    def edit(self, text: str) -> Iterator[dict]:
        """Change the plan with a sentence. It goes to the Edit Router."""
        self.bb.write(edit_directive=None, assistant_reply=None, iteration_count=0)
        self.bus.post(MsgType.EDIT, TRAVELLER, "edit_router", text[:90], text=text)
        return self._drain()

    def run_to_end(self, events: Iterator[dict]) -> None:
        for _ in events:
            pass

    # -- the loop --------------------------------------------------------------------

    def _drain(self) -> Iterator[dict]:
        while (msg := self.bus.pop()) is not None:
            yield msg.public()
            target: Any = self.agents.get(msg.recipient) or (self.environment if msg.recipient == "environment" else None)
            if target is None:  # addressed to the traveller: shown, not processed
                continue
            yield {"type": "step_start", "agent": target.name, "message": f"{target.name} started..."}
            result = target.handle(msg, self.bb.view(target))
            self.bb.commit(target, result.updates)
            for post in result.posts:
                self.bus.post(post.type, target.name, post.recipient, post.summary, **post.payload)
            self.runs[target.name] += 1
            yield {"type": "step_complete", "agent": target.name, "message": result.message or f"{target.name} completed.", "meta": result.meta()}

    # -- what came out ---------------------------------------------------------------

    def stats(self) -> dict:
        agent_runs = sum(n for name, n in self.runs.items() if name != "environment")
        return {
            "agent_runs": agent_runs,
            "runs_by_agent": dict(self.runs),
            "messages": len(self.bus.log),
            "message_counts": self.bus.counts(),
            "field_checks": self.bb["observed"].checks,
            "iterations": self.bb["iteration_count"],
        }

    def snapshot(self) -> dict:
        return {
            **serialize_final_state(self.bb.data),
            "messages": [m.public() for m in self.bus.log],
            "stats": self.stats(),
            "strategy": self.strategy.name,
            "seed": self.seed,
            "uncertainty": self.uncertainty,
        }
