"""What an agent reports about its own run. The UI shows these as chips next to the step."""
from __future__ import annotations


def trace(tools: list[str] | None = None, algorithms: list[str] | None = None, note: str = "") -> dict:
    """tools = the deterministic tools the agent called; algorithms = the search/CSP it ran."""
    return {"tools": list(tools or []), "algorithms": list(algorithms or []), "note": note}
