"""SQLite store for finalized itineraries (used after a process restart)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base

from config import BACKEND_DIR

_DB_PATH = BACKEND_DIR / "odyssey.db"
engine = create_engine(f"sqlite:///{_DB_PATH}", future=True)
Base = declarative_base()


class SavedItinerary(Base):
    __tablename__ = "itineraries"

    thread_id = Column(String, primary_key=True)
    payload_json = Column(Text, nullable=False)
    score = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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
