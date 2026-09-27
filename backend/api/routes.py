from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sse import serialize_final_state, stream_graph_run
from models.schemas import Disruption, DisruptionType
from orchestration.graph import get_graph
from orchestration.state import initial_state
from tools import world

router = APIRouter(prefix="/api")

_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
_pending: dict[str, dict] = {}  # input for the next stream of a thread (a new plan or an edit)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _current(thread_id: str) -> dict:
    """The plan a thread has now (LangGraph's in-memory checkpoint), or 404."""
    values = get_graph().get_state(_config(thread_id)).values
    if not values:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
    return values


class PlanRequest(BaseModel):
    message: str
    thread_id: str | None = None


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


@router.post("/plan")
async def create_plan(req: PlanRequest):
    """Start a planning session. The client then opens GET /api/plan/{thread_id}/stream."""
    thread_id = req.thread_id or str(uuid.uuid4())
    _pending[thread_id] = initial_state(req.message)
    return {"thread_id": thread_id}


@router.post("/revise")
async def revise_plan(req: ReviseRequest):
    """Change an existing plan with a sentence. The Edit Router decides which agents re-run; the client
    then opens the stream."""
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message")
    _pending[req.thread_id] = {
        **_current(req.thread_id),
        "edit_request": req.message.strip(),
        "edit_directive": None,
        "assistant_reply": None,
        "iteration_count": 0,
    }
    return {"thread_id": req.thread_id, "status": "queued"}


@router.get("/plan/{thread_id}/stream")
async def stream_plan(thread_id: str):
    """SSE stream of one run: a new plan, an edit, or the replan after a disruption."""
    graph = get_graph()
    config = _config(thread_id)
    input_state = _pending.pop(thread_id, None)
    if input_state is None:
        _current(thread_id)  # 404 if the thread does not exist; otherwise resume (a disruption was injected)
    return StreamingResponse(stream_graph_run(graph, thread_id, input_state, config), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/disrupt")
async def inject_disruption(req: DisruptRequest):
    """Inject a disruption. The client then reconnects to the stream, which starts at the Critic."""
    values = _current(req.thread_id)
    disruption = Disruption(type=req.type, target=req.target, description=req.description, day=req.day, new_budget_inr=req.new_budget_inr)
    get_graph().update_state(
        _config(req.thread_id),
        {"disruptions": [*values.get("disruptions", []), disruption], "iteration_count": 0, "edit_directive": None, "assistant_reply": None},
        as_node="itinerary_architect",  # the next node to run is the Critic
    )
    return {"thread_id": req.thread_id, "status": "injected"}


@router.get("/itinerary/{thread_id}")
async def get_itinerary(thread_id: str):
    """The current plan of a thread (used when the plan page is reloaded)."""
    return serialize_final_state(_current(thread_id))


@router.get("/health")
async def health():
    """Server state. Planning is deterministic and offline: no model, no external API."""
    return {
        "status": "ok",
        "model": "none (deterministic rules and search)",
        "regions": [{"region": name, "cities": len(data["cities"])} for name, data in world.regions().items()],
    }
