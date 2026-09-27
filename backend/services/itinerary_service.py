"""SQLite store for finalized itineraries and the graph state behind them (so a plan can be edited after a restart)."""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import Column, DateTime, Float, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base

from config import BACKEND_DIR, DB_PATH

logger = logging.getLogger("odyssey.itinerary_service")

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
_serde = JsonPlusSerializer()
Base = declarative_base()


class SavedItinerary(Base):
    __tablename__ = "itineraries"

    thread_id = Column(String, primary_key=True)
    payload_json = Column(Text, nullable=False)
    score = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SavedState(Base):
    __tablename__ = "states"

    thread_id = Column(String, primary_key=True)
    state_type = Column(String, nullable=False)
    state_blob = Column(LargeBinary, nullable=False)


Base.metadata.create_all(engine)


def save_itinerary(thread_id: str, payload: dict) -> None:
    score = payload.get("score") or 0.0
    with Session(engine) as session:
        row = session.get(SavedItinerary, thread_id)
        if row is None:
            row = SavedItinerary(thread_id=thread_id)
            session.add(row)
        row.payload_json = json.dumps(payload, default=str)
        row.score = float(score or 0.0)
        row.updated_at = datetime.now(timezone.utc)
        session.commit()


def load_itinerary(thread_id: str) -> dict | None:
    with Session(engine) as session:
        row = session.get(SavedItinerary, thread_id)
        if row is None:
            return None
        try:
            return json.loads(row.payload_json)
        except json.JSONDecodeError:
            return None


def save_state(thread_id: str, values: dict) -> None:
    kind, blob = _serde.dumps_typed({k: v for k, v in values.items() if k != "agent_messages"})
    with Session(engine) as session:
        session.merge(SavedState(thread_id=thread_id, state_type=kind, state_blob=blob))
        session.commit()


def save_plan(thread_id: str, payload: dict, values: dict) -> None:
    """Save the itinerary and its editable graph state together, in one transaction.

    Saving them as two separate commits (the original shape of this module) left a window
    where a crash or a `--reload` restart between the two writes could leave a thread with a
    perfectly good itinerary but no state row — that plan renders fine on load but every edit
    404s with a misleading "unknown thread". One commit makes that window impossible.
    """
    score = payload.get("score") or 0.0
    kind, blob = _serde.dumps_typed({k: v for k, v in values.items() if k != "agent_messages"})
    with Session(engine) as session:
        row = session.get(SavedItinerary, thread_id)
        if row is None:
            row = SavedItinerary(thread_id=thread_id)
            session.add(row)
        row.payload_json = json.dumps(payload, default=str)
        row.score = float(score or 0.0)
        row.updated_at = datetime.now(timezone.utc)
        session.merge(SavedState(thread_id=thread_id, state_type=kind, state_blob=blob))
        session.commit()


def load_state(thread_id: str) -> dict | None:
    with Session(engine) as session:
        row = session.get(SavedState, thread_id)
        if row is None:
            return None
        try:
            return _serde.loads_typed((row.state_type, row.state_blob))
        except Exception:  # noqa: BLE001 — a corrupt/incompatible blob is "no saved state", not a 500
            logger.exception("failed to deserialize saved state for thread %s", thread_id)
            return None


SAMPLE_PATH = BACKEND_DIR / "sample_trip.json"


def copy_sample(thread_id: str) -> bool:
    """Store the saved sample trip under a new thread, so it opens and can be edited like any plan."""
    if not SAMPLE_PATH.exists():
        return False
    sample = json.loads(SAMPLE_PATH.read_text())
    payload = sample["payload"]
    with Session(engine) as session:  # one transaction: see save_plan's note on why this must not be two commits
        row = session.get(SavedItinerary, thread_id)
        if row is None:
            row = SavedItinerary(thread_id=thread_id)
            session.add(row)
        row.payload_json = json.dumps(payload, default=str)
        row.score = float(payload.get("score") or 0.0)
        row.updated_at = datetime.now(timezone.utc)
        session.merge(SavedState(thread_id=thread_id, state_type=sample["state_type"], state_blob=base64.b64decode(sample["state_blob"])))
        session.commit()
    return True
