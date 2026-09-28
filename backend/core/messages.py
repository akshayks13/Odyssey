"""The message protocol. Agents never call each other: they post addressed messages on a bus.

Every arrow in the system is one of these messages:

    traveller ── REQUEST ────────► trip_analyst
    trip_analyst ── SPEC_READY ──► destination_agent          (or NEED_INFO ► traveller)
    destination_agent ── CITIES_READY ► mobility_agent
    mobility_agent ── ROUTE_READY ───► budget_agent
    budget_agent ── BUDGET_READY ────► itinerary_architect
    itinerary_architect ── SCHEDULE_READY ► environment       (field check, when observing)
    environment ── FIELD_REPORT ─────► critic_replanner      (or straight from the Architect when not observing)
    critic_replanner ── REPLAN ──────► the one agent that owns the issue
    critic_replanner ── ACCEPT / BEST_EFFORT ► traveller
    traveller ── DISRUPTION ─────────► critic_replanner
    traveller ── EDIT ───────────────► edit_router
    edit_router ── RERUN ────────────► the earliest agent whose inputs changed   (or ANSWER ► traveller)
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MsgType(str, Enum):
    REQUEST = "REQUEST"
    SPEC_READY = "SPEC_READY"
    NEED_INFO = "NEED_INFO"
    CITIES_READY = "CITIES_READY"
    ROUTE_READY = "ROUTE_READY"
    BUDGET_READY = "BUDGET_READY"
    SCHEDULE_READY = "SCHEDULE_READY"
    FIELD_REPORT = "FIELD_REPORT"
    REPLAN = "REPLAN"
    RERUN = "RERUN"
    ACCEPT = "ACCEPT"
    BEST_EFFORT = "BEST_EFFORT"
    DISRUPTION = "DISRUPTION"
    EDIT = "EDIT"
    ANSWER = "ANSWER"


TRAVELLER = "traveller"  # the human: messages addressed to it are shown, not processed


@dataclass
class Message:
    id: int
    type: MsgType
    sender: str
    recipient: str
    summary: str = ""  # one line for the UI log
    payload: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict:
        """What the UI gets: who told whom what. The payload stays inside the system."""
        return {"type": "message", "id": self.id, "kind": self.type.value, "from": self.sender, "to": self.recipient, "summary": self.summary}


class MessageBus:
    """A FIFO queue plus a log of everything ever sent (the log is what the demo shows)."""

    def __init__(self) -> None:
        self.log: list[Message] = []
        self._queue: deque[Message] = deque()
        self._next_id = 1

    def post(self, type: MsgType, sender: str, recipient: str, summary: str = "", **payload: Any) -> Message:
        msg = Message(self._next_id, type, sender, recipient, summary, payload)
        self._next_id += 1
        self.log.append(msg)
        self._queue.append(msg)
        return msg

    def pop(self) -> Message | None:
        return self._queue.popleft() if self._queue else None

    def counts(self) -> dict[str, int]:
        return dict(Counter(m.type.value for m in self.log))
