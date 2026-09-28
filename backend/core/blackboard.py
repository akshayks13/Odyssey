"""The shared belief state, with access control.

Agents share one `Blackboard`, but each may only read the keys it declares in `reads` and write the keys it
declares in `writes`. `view()` hands an agent a copy limited to its reads, and `commit()` refuses a write
outside its writes. That is what "each agent owns its own part of the state" means in the code, and the
tests check it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from models.schemas import EditLocks, ObservedFacts

if TYPE_CHECKING:
    from agents.base import Agent


class AccessError(RuntimeError):
    """An agent tried to write something it does not own."""


def initial_state(raw_input: str = "") -> dict[str, Any]:
    return {
        "raw_input": raw_input,
        "trip_spec": None,
        "candidate_destinations": [],
        "candidate_activities": {},
        "selected_destinations": [],
        "route": None,
        "budget_breakdown": None,
        "accommodation_options": [],
        "excluded_activity_ids": [],
        "stay_plan": [],
        "draft_itinerary": None,
        "final_itinerary": None,
        "validation_report": None,
        "replan_directives": [],
        "iteration_count": 0,
        "conflicts": [],
        "disruptions": [],
        "observed": ObservedFacts(),
        "edit_directive": None,
        "edit_locks": EditLocks(),
        "assistant_reply": None,
        "optimization_score": 0.0,
    }


class View(dict):
    """What an agent sees: only the keys it declared. Asking for any other key raises `AccessError`, even through `.get()`."""

    def __init__(self, owner: str, data: dict[str, Any]) -> None:
        super().__init__(data)
        self.owner = owner

    def __missing__(self, key: str) -> Any:
        raise AccessError(f"{self.owner} read {key!r}, which it did not declare in `reads`")

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            raise AccessError(f"{self.owner} read {key!r}, which it did not declare in `reads`")
        return super().get(key, default)


class Blackboard:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data if data is not None else initial_state()

    def view(self, agent: "Agent") -> View:
        """A snapshot of exactly the keys this agent said it reads."""
        return View(agent.name, {key: self.data.get(key) for key in agent.reads})

    def commit(self, agent: "Agent", updates: dict[str, Any]) -> None:
        illegal = set(updates) - set(agent.writes)
        if illegal:
            raise AccessError(f"{agent.name} may not write {sorted(illegal)}; it owns {sorted(agent.writes)}")
        self.data.update(updates)

    def write(self, **updates: Any) -> None:
        """The runtime and the traveller write here directly (a new request, a reported event)."""
        self.data.update(updates)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
