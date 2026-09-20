"""Map LangGraph `astream` chunks onto SSE events for the timeline UI."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from models.schemas import BudgetBreakdown, Itinerary, Route, ValidationReport
from services.itinerary_service import save_itinerary

_STREAM_MODES = ["messages", "updates", "custom"]
_KNOWN_AGENTS = {
    "trip_analyst",
    "destination_agent",
    "mobility_agent",
    "budget_agent",
    "itinerary_architect",
    "critic_replanner",
}


def sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def serialize_final_state(values: dict) -> dict:
    itinerary: Itinerary | None = values.get("final_itinerary") or values.get("draft_itinerary")
    budget: BudgetBreakdown | None = values.get("budget_breakdown")
    route: Route | None = values.get("route")
    validation: ValidationReport | None = values.get("validation_report")

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
    }


def _unpack(item: Any) -> tuple[str, Any, tuple]:
    """Normalize astream yields across stream_mode list / subgraphs combos."""
    if isinstance(item, tuple):
        if len(item) == 3:
            ns, mode, data = item
            if mode in _STREAM_MODES or mode in {"values", "debug"}:
                return str(mode), data, ns if isinstance(ns, tuple) else ()
            return "messages", item, ()
        if len(item) == 2:
            a, b = item
            if a in _STREAM_MODES or a in {"values", "debug"}:
                return str(a), b, ()
            return "messages", item, ()
    if isinstance(item, dict):
        return "updates", item, ()
    return "custom", item, ()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content or "")


def _compact_node_data(node_output: dict | None) -> dict:
    if not node_output:
        return {}
    data: dict[str, Any] = {}
    if node_output.get("draft_itinerary") is not None:
        itinerary = node_output["draft_itinerary"]
        data["draft_itinerary"] = itinerary.model_dump() if hasattr(itinerary, "model_dump") else itinerary
    if node_output.get("budget_breakdown") is not None:
        budget = node_output["budget_breakdown"]
        data["budget"] = budget.model_dump() if hasattr(budget, "model_dump") else budget
    if node_output.get("route") is not None:
        route = node_output["route"]
        data["route"] = route.model_dump() if hasattr(route, "model_dump") else route
    return data


async def stream_graph_run(graph: Any, thread_id: str, input_state: dict | None, config: dict) -> AsyncGenerator[str, None]:
    """Stream one graph run as SSE events."""
    yield sse_event({"type": "init", "thread_id": thread_id})
    started: set[str] = set()
    try:
        astream_fn = graph.astream
        try:
            stream = astream_fn(
                input_state,
                config=config,
                version="v2",
                stream_mode=_STREAM_MODES,
                subgraphs=True,
            )
        except TypeError:
            stream = astream_fn(
                input_state,
                config=config,
                stream_mode=_STREAM_MODES,
                subgraphs=True,
            )

        async for item in stream:
            mode, data, _ns = _unpack(item)

            if mode == "updates" and isinstance(data, dict):
                for node_name, node_output in data.items():
                    if node_name.startswith("__"):
                        continue
                    if node_name not in started:
                        started.add(node_name)
                        yield sse_event(
                            {
                                "type": "step_start",
                                "agent": node_name,
                                "message": f"{node_name} started...",
                            }
                        )
                    messages = (node_output or {}).get("agent_messages", []) if isinstance(node_output, dict) else []
                    message = messages[-1] if messages else f"{node_name} completed."
                    yield sse_event(
                        {
                            "type": "step_complete",
                            "agent": node_name,
                            "message": message,
                            "data": _compact_node_data(node_output if isinstance(node_output, dict) else None),
                        }
                    )

            elif mode == "messages":
                message, metadata = (data if isinstance(data, tuple) and len(data) == 2 else (data, {}))
                meta = metadata if isinstance(metadata, dict) else {}
                agent = meta.get("langgraph_node") or meta.get("checkpoint_ns") or ""
                name = getattr(message, "name", None) or getattr(message, "type", "") or ""
                text = _message_text(message)
                tool_calls = getattr(message, "tool_calls", None) or []
                if agent and agent in _KNOWN_AGENTS and agent not in started:
                    started.add(agent)
                    yield sse_event({"type": "step_start", "agent": agent, "message": f"{agent} started..."})
                if tool_calls:
                    for call in tool_calls:
                        tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "tool")
                        yield sse_event(
                            {
                                "type": "tool_result",
                                "agent": agent or None,
                                "tool": tool_name,
                                "data": call.get("args") if isinstance(call, dict) else {},
                            }
                        )
                elif name == "tool" or getattr(message, "type", "") == "tool":
                    yield sse_event(
                        {
                            "type": "tool_result",
                            "agent": agent or None,
                            "tool": getattr(message, "name", "tool"),
                            "data": {"content": text[:500]},
                        }
                    )
                elif text.strip():
                    yield sse_event({"type": "message", "agent": agent or None, "message": text[:500]})

            elif mode == "custom":
                payload = data if isinstance(data, dict) else {"data": data}
                yield sse_event({"type": "custom", **payload})

        final_values = graph.get_state(config).values
        done_payload = serialize_final_state(final_values)
        save_itinerary(thread_id, done_payload)
        yield sse_event({"type": "done", **done_payload})
    except Exception as exc:  # noqa: BLE001 — surface any failure to the client as an SSE error event
        yield sse_event({"type": "error", "message": str(exc)})
