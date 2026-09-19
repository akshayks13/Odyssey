"""FastAPI application entrypoint.

Run locally with:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import CORS_ORIGINS
from orchestration.graph import get_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Compile the LangGraph StateGraph once at startup and reuse it across
    # every request (MemorySaver keeps per-thread_id state in-process).
    get_graph()
    yield


app = FastAPI(title="Odyssey — Multi-Agent Travel Planner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"name": "Odyssey", "status": "running", "docs": "/docs"}
