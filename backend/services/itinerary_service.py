"""SQLite store for finalized itineraries and the graph state behind them (so a plan can be edited after a restart)."""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import Column, DateTime, Float, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base

from config import BACKEND_DIR, DB_PATH

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


def load_state(thread_id: str) -> dict | None:
    with Session(engine) as session:
        row = session.get(SavedState, thread_id)
        return _serde.loads_typed((row.state_type, row.state_blob)) if row else None


SAMPLE_PATH = BACKEND_DIR / "sample_trip.json"


def copy_sample(thread_id: str) -> bool:
    """Store the saved sample trip under a new thread, so it opens and can be edited like any plan."""
    if not SAMPLE_PATH.exists():
        return False
    sample = json.loads(SAMPLE_PATH.read_text())
    save_itinerary(thread_id, sample["payload"])
    with Session(engine) as session:
        session.merge(SavedState(thread_id=thread_id, state_type=sample["state_type"], state_blob=base64.b64decode(sample["state_blob"])))
        session.commit()
    return True
