from __future__ import annotations

import uuid
import zlib
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sse import stream_events
from core.compare import compare_strategies
from core.environment import LEVELS
from core.strategies import STRATEGIES
from core.team import Team
from models.schemas import Disruption, DisruptionType
from tools import world

router = APIRouter(prefix="/api")

_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
_teams: dict[str, Team] = {}  # one team per thread, kept in memory while the server runs
_pending: dict[str, Iterator[dict]] = {}  # the run each thread will do when its stream is opened


def _team(thread_id: str) -> Team:
    team = _teams.get(thread_id)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
    return team


class PlanRequest(BaseModel):
    message: str
    thread_id: str | None = None
    strategy: str = "full"
    uncertainty: str = "normal"  # off | normal | high: how often the field surprises the plan
    seed: int | None = None  # the field's random seed; by default derived from the request, so a request always meets the same field


class ReviseRequest(BaseModel):
    thread_id: str
    message: str


class DisruptRequest(BaseModel):
    thread_id: str
    type: DisruptionType
    target: str
    description: str
    day: int | None = None
    new_budget_inr: float | None = None


class CompareRequest(BaseModel):
    thread_id: str


@router.post("/plan")
async def create_plan(req: PlanRequest):
    """Start a planning session. The client then opens GET /api/plan/{thread_id}/stream."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"strategy must be one of {sorted(STRATEGIES)}")
    if req.uncertainty not in LEVELS:
        raise HTTPException(status_code=422, detail=f"uncertainty must be one of {sorted(LEVELS)}")
    thread_id = req.thread_id or str(uuid.uuid4())
    seed = req.seed if req.seed is not None else zlib.crc32(req.message.encode())
    team = _teams[thread_id] = Team(req.strategy, seed, req.uncertainty)
    _pending[thread_id] = team.plan(req.message)
    return {"thread_id": thread_id}


@router.post("/revise")
async def revise_plan(req: ReviseRequest):
    """Change an existing plan with a sentence. The Edit Router decides which agents re-run; the client then opens the stream."""
    team = _team(req.thread_id)
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message")
    _pending[req.thread_id] = team.edit(req.message.strip())
    return {"thread_id": req.thread_id, "status": "queued"}


@router.get("/plan/{thread_id}/stream")
async def stream_plan(thread_id: str):
    """SSE stream of one run: a new plan, an edit, or the replan after a disruption. With nothing queued it just returns the plan."""
    team = _team(thread_id)
    events = _pending.pop(thread_id, iter(()))
    return StreamingResponse(stream_events(team, thread_id, events), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/disrupt")
async def inject_disruption(req: DisruptRequest):
    """Report an event. The client then reconnects to the stream, which starts at the Critic."""
    team = _team(req.thread_id)
    disruption = Disruption(type=req.type, target=req.target, description=req.description, day=req.day, new_budget_inr=req.new_budget_inr)
    _pending[req.thread_id] = team.disrupt(disruption)
    return {"thread_id": req.thread_id, "status": "injected"}


@router.get("/itinerary/{thread_id}")
async def get_itinerary(thread_id: str):
    """The current plan of a thread (used when the plan page is reloaded)."""
    return _team(thread_id).snapshot()


@router.post("/compare")
async def compare(req: CompareRequest):
    """Run every strategy on this thread's request, field and reported events, and score each against what really happened."""
    team = _team(req.thread_id)
    if not team.request:
        raise HTTPException(status_code=409, detail="Nothing to compare yet: make a plan first")
    return {
        "request": team.request,
        "seed": team.seed,
        "uncertainty": team.uncertainty,
        "reported": len(team.reported),
        "strategies": compare_strategies(team.request, team.seed, team.uncertainty, team.reported),
    }


@router.get("/agents")
async def agents():
    """The six agents and the edit router: role, architecture, PEAS, and what each reads and writes."""
    team = Team()
    return [
        {"name": a.name, "label": a.label, "role": a.role, "architecture": a.architecture, "peas": a.peas, "reads": list(a.reads), "writes": list(a.writes)}
        for a in team.agents.values()
    ]


@router.get("/health")
async def health():
    """Server state. Planning is deterministic and offline: no model, no external API."""
    return {
        "status": "ok",
        "model": "none (deterministic rules and search)",
        "regions": [{"region": name, "cities": len(data["cities"])} for name, data in world.regions().items()],
        "strategies": {name: s.label for name, s in STRATEGIES.items()},
    }
