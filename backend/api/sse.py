"""Stream a LangGraph run to the UI as server-sent events: one event per agent step, then the plan."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from models.schemas import BudgetBreakdown, Itinerary, Route, ValidationReport

logger = logging.getLogger("odyssey.sse")

def sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _assumptions(spec) -> list[str]:
    """What the Analyst filled in because the traveller didn't say, with the values used."""
    if not spec:
        return []
    text = {
        "duration": f"trip length: {spec.duration_days} days",
        "travellers": f"travellers: {spec.travellers}",
        "budget": f"budget: ₹{spec.budget_inr:,.0f}",
        "start_date": f"start date: {spec.start_date}",
    }
    return [text[n] for n in spec.needs_clarification if n in text]


def serialize_final_state(values: dict) -> dict:
    itinerary: Itinerary | None = values.get("final_itinerary") or values.get("draft_itinerary")
    budget: BudgetBreakdown | None = values.get("budget_breakdown")
    route: Route | None = values.get("route")
    validation: ValidationReport | None = values.get("validation_report")
    spec = values.get("trip_spec")
    directive = values.get("edit_directive")

    return {
        "itinerary": itinerary.model_dump() if itinerary else None,
        "budget": budget.model_dump() if budget else None,
        "route": route.model_dump() if route else None,
        "valid": validation.valid if validation else None,
        "issues": [i.model_dump() for i in validation.issues] if validation else [],
        "score": itinerary.optimization_score if itinerary else None,
        "iteration_count": values.get("iteration_count", 0),
        "selected_destinations": [d.model_dump() for d in values.get("selected_destinations", [])],
        "accommodation_options": [h.model_dump() for h in values.get("accommodation_options", [])],
        "trip": {"duration_days": spec.duration_days, "start_date": spec.start_date, "travellers": spec.travellers} if spec else None,
        "assumptions": _assumptions(spec),
        "reply": values.get("assistant_reply"),
        "summary": directive.summary if directive and directive.intent == "modify" else None,
        "reran_from": directive.entry if directive and directive.intent == "modify" else None,
    }


async def stream_graph_run(graph: Any, thread_id: str, input_state: dict | None, config: dict) -> AsyncGenerator[str, None]:
    """init, then step_start / step_complete for every agent as it finishes, then done (or error)."""
    yield sse_event({"type": "init", "thread_id": thread_id})
    try:
        async for chunk in graph.astream(input_state, config=config, stream_mode="updates"):
            for agent, output in chunk.items():
                if agent.startswith("__") or not isinstance(output, dict):
                    continue
                messages = output.get("agent_messages") or []
                yield sse_event({"type": "step_start", "agent": agent, "message": f"{agent} started..."})
                yield sse_event({
                    "type": "step_complete",
                    "agent": agent,
                    "message": messages[-1] if messages else f"{agent} completed.",
                    "meta": output.get("agent_meta") or {},
                })
        yield sse_event({"type": "done", **serialize_final_state(graph.get_state(config).values)})
    except Exception as exc:  # noqa: BLE001 — surface any failure to the client as an SSE error event
        logger.exception("run failed for thread %s", thread_id)
        yield sse_event({"type": "error", "message": str(exc), "thread_id": thread_id})
