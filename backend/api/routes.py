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
)
from llm import llm_provider_name, providers_status
from models.schemas import Disruption, DisruptionType
from orchestration.graph import get_graph
from orchestration.state import initial_state
from services.itinerary_service import copy_sample, load_itinerary, load_state

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
    """Start a LangGraph planning session. Returns `thread_id`; the client
    then opens GET /api/plan/{thread_id}/stream for SSE."""
    thread_id = req.thread_id or str(uuid.uuid4())
    _pending[thread_id] = initial_state(req.message)
    return {"thread_id": thread_id}


@router.post("/sample")
async def open_sample():
    """Open the saved sample trip as a new thread. Needs no model, so a demo works when none is available."""
    thread_id = str(uuid.uuid4())
    if not copy_sample(thread_id):
        raise HTTPException(status_code=404, detail="No sample trip is saved")
    return {"thread_id": thread_id}


@router.post("/revise")
async def revise_plan(req: ReviseRequest):
    """Change an existing plan with a sentence. The Edit Router decides which agents re-run; the
    client then opens the stream. Works from the stored state, so it survives a server restart."""
    stored = load_state(req.thread_id)
    if stored is None or not req.message.strip():
        raise HTTPException(status_code=404 if stored is None else 422, detail="Unknown thread_id or empty message")
    _pending[req.thread_id] = {
        **stored,
        "edit_request": req.message.strip(),
        "edit_directive": None,
        "assistant_reply": None,
        "iteration_count": 0,
    }
    return {"thread_id": req.thread_id, "status": "queued"}


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
    if not existing_state.values:  # not in memory (server restarted, or the sample trip): resume from the saved plan
        stored = load_state(req.thread_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id: {req.thread_id}")
        graph.update_state(config, stored, as_node="itinerary_architect")
        existing_state = graph.get_state(config)

    disruption = Disruption(
        type=req.type,
        target=req.target,
        description=req.description,
        day=req.day,
        new_budget_inr=req.new_budget_inr,
    )
    existing_disruptions = existing_state.values.get("disruptions", [])

    graph.update_state(
        config,
        {"disruptions": [*existing_disruptions, disruption], "iteration_count": 0, "edit_directive": None, "assistant_reply": None},
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
    """Server state. `models.state` says what the models are really doing — a configured key is not a
    working key, so it reads "configured_but_untried" until one has actually answered."""
    models = providers_status()
    return {
        "status": "ok",
        "model": llm_provider_name() or "unavailable",
        "models": models,
        "langsmith": {"enabled": LANGSMITH_TRACING, "project": LANGSMITH_PROJECT if LANGSMITH_TRACING else None},
        "apis": {
            "mapbox": "key set" if MAPBOX_API_KEY else "absent (OpenStreetMap only)",
            "foursquare": "key set but unused" if FOURSQUARE_API_KEY else "unused",
            "weather": "open-meteo",
            "hotels_flights": "llm",
        },
    }
