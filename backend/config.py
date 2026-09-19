"""Central configuration — reads environment variables with safe defaults.

All external API calls are optional: if a key is missing, the corresponding
tool transparently falls back to `data/kerala_seed.json` so the demo never
breaks on a missing/invalid credential.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend/ directory if present
load_dotenv(Path(__file__).parent / ".env")

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
SEED_DATA_PATH = PROJECT_ROOT / "data" / "kerala_seed.json"

# --- LLM ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Tests / CI set this so the suite never bills a live model.
LLM_DISABLED = os.getenv("ODYSSEY_DISABLE_LLM", "").lower() in {"1", "true", "yes"}

# --- LangSmith observability -------------------------------------------
# LangGraph/LangChain auto-trace every node + tool call once these are set.
# Both LANGSMITH_* and the older LANGCHAIN_* names are populated because
# some langchain-core versions still read the latter.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "odyssey"
LANGSMITH_TRACING = bool(LANGSMITH_API_KEY) and os.getenv("LANGSMITH_TRACING", "true").lower() not in {"0", "false", "no"}

if LANGSMITH_API_KEY and LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# --- External travel APIs (all optional) ---------------------------------
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY", "")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY", "")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")

# --- Replanning ---------------------------------------------------------
MAX_REPLAN_ITERATIONS = int(os.getenv("MAX_REPLAN_ITERATIONS", "3"))

# --- Server ---------------------------------------------------------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")


def has_key(name: str) -> bool:
    return bool(globals().get(name, ""))
