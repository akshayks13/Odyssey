"""What every agent is: a role, a PEAS description, the state it reads and writes, and a `handle` method.

An agent reacts to one message at a time. It receives a `view` of the blackboard (only the keys it declared in
`reads`) and returns a `Result`: the blackboard updates it owns, the messages it wants to send, and a short
account of what it did (shown in the demo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.messages import Message, MsgType
from core.strategies import Strategy


@dataclass
class Post:
    """A message an agent wants sent."""

    type: MsgType
    recipient: str
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    updates: dict[str, Any] = field(default_factory=dict)
    posts: list[Post] = field(default_factory=list)
    message: str = ""  # one sentence for the timeline
    tools: list[str] = field(default_factory=list)  # the deterministic tools called
    algorithms: list[str] = field(default_factory=list)  # the search / CSP / rules run
    note: str = ""

    def meta(self) -> dict:
        return {"tools": self.tools, "algorithms": self.algorithms, "note": self.note}


class Agent:
    name: str = ""
    label: str = ""
    role: str = ""
    architecture: str = ""  # simple reflex | goal-based | utility-based
    peas: dict[str, str] = {}
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy

    def handle(self, msg: Message, view: dict[str, Any]) -> Result:  # pragma: no cover - interface
        raise NotImplementedError
