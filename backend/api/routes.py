from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.sse import stream_graph_run, serialize_final_state
from config import (
    FOURSQUARE_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    MAPBOX_API_KEY,
    OPENWEATHER_API_KEY,
)
from llm import llm_provider_name
from models.schemas import Disruption, DisruptionType
from orchestration.graph import get_graph
from orchestration.state import initial_state
from services.itinerary_service import load_itinerary

router = APIRouter(prefix="/api")

_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"}
_pending: dict[str, dict] = {}


def _graph_config(thread_id: str, *tags: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"odyssey:{thread_id[:8]}",
        "tags": ["odyssey", *tags],
        "metadata": {"thread_id": thread_id},
    }


class PlanRequest(BaseModel):
    message: str
    thread_id: str | None = None


class DisruptRequest(BaseModel):
    thread_id: str
    type: DisruptionType
    target: str
    description: str
    day: int | None = None
    new_budget_inr: float | None = None


@router.post("/plan")
async def create_plan(req: PlanRequest):
    """Start a LangGraph planning session. Returns `thread_id`; the client
    then opens GET /api/plan/{thread_id}/stream for SSE."""
    thread_id = req.thread_id or str(uuid.uuid4())
    _pending[thread_id] = initial_state(req.message)
    return {"thread_id": thread_id}


@router.get("/plan/{thread_id}/stream")
async def stream_plan(thread_id: str):
    """SSE stream of one graph run.

    Uses `graph.astream(version="v2", stream_mode=["messages","updates","custom"], subgraphs=True)`
    when the installed LangGraph accepts `version`; otherwise the same
    stream_mode list on `astream` (LangGraph 0.2.60).
    """
    graph = get_graph()
    config = _graph_config(thread_id, "stream")
    pending = _pending.pop(thread_id, None)
    existing = graph.get_state(config)
    if pending is not None:
        input_state = pending
    elif existing.values:
        input_state = None
    else:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")

    return StreamingResponse(
        stream_graph_run(graph, thread_id, input_state, config),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/disrupt")
async def inject_disruption(req: DisruptRequest):
    """Inject a live disruption into an existing session. The client then
    reconnects via GET /api/plan/{thread_id}/stream to run the targeted replan."""
    graph = get_graph()
    config = _graph_config(req.thread_id, "disrupt")

    existing_state = graph.get_state(config)
    if not existing_state.values:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {req.thread_id}")

    disruption = Disruption(
        type=req.type,
        target=req.target,
        description=req.description,
        day=req.day,
        new_budget_inr=req.new_budget_inr,
    )
    existing_disruptions = existing_state.values.get("disruptions", [])

    # `as_node="itinerary_architect"` (a direct predecessor of the Critic in
    # the graph) makes LangGraph resume execution AT the Critic with this
    # new state merged in, so the Critic actually re-evaluates and routes.
    graph.update_state(
        config,
        {"disruptions": [*existing_disruptions, disruption], "iteration_count": 0},
        as_node="itinerary_architect",
    )
    return {"thread_id": req.thread_id, "status": "injected"}


@router.get("/itinerary/{thread_id}")
async def get_itinerary(thread_id: str):
    """Fetch the current finalized (or best-effort) state for a session —
    used on page reload or for a non-streaming poll."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if state.values:
        return serialize_final_state(state.values)
    saved = load_itinerary(thread_id)
    if saved:
        return saved
    raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")


@router.get("/health")
async def health():
    llm = llm_provider_name()
    return {
        "status": "ok",
        "llm": llm,
        "mode": "llm_tool_calling" if llm else "offline_heuristics_seed",
        "langsmith": {"enabled": LANGSMITH_TRACING, "project": LANGSMITH_PROJECT if LANGSMITH_TRACING else None},
        "apis": {
            "mapbox": bool(MAPBOX_API_KEY),
            "foursquare": bool(FOURSQUARE_API_KEY),
            "weather": bool(OPENWEATHER_API_KEY),
            "hotels_flights": "gemini_generated",
        },
    }
