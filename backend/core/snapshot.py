"""The current plan as plain data, in the shape the UI reads."""
from __future__ import annotations

from typing import Any

from models.schemas import BudgetBreakdown, Itinerary, Route, ValidationReport


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


def serialize_final_state(values: dict[str, Any]) -> dict:
    itinerary: Itinerary | None = values.get("final_itinerary") or values.get("draft_itinerary")
    budget: BudgetBreakdown | None = values.get("budget_breakdown")
    route: Route | None = values.get("route")
    validation: ValidationReport | None = values.get("validation_report")
    spec = values.get("trip_spec")
    directive = values.get("edit_directive")
    observed = values.get("observed")

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
        "field_checks": observed.checks if observed else 0,
        "conflicts": [c.model_dump() for c in values.get("conflicts", [])],
    }
