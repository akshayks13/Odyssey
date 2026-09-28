"""Stream a run to the UI as server-sent events: every message on the bus, one step per agent, then the plan."""
from __future__ import annotations

import json
import logging
from typing import Iterator

from core.team import Team

logger = logging.getLogger("odyssey.sse")


def sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def stream_events(team: Team, thread_id: str, events: Iterator[dict]) -> Iterator[str]:
    """init, then message / step_start / step_complete as the agents work, then done (or error)."""
    yield sse_event({"type": "init", "thread_id": thread_id})
    try:
        for event in events:
            yield sse_event(event)
        yield sse_event({"type": "done", **team.snapshot()})
    except Exception as exc:  # noqa: BLE001 — surface any failure to the client as an SSE error event
        logger.exception("run failed for thread %s", thread_id)
        yield sse_event({"type": "error", "message": str(exc), "thread_id": thread_id})
