"""Settings. The planner is offline and deterministic, so there are no API keys."""
from __future__ import annotations

import os

# --- Replanning ---------------------------------------------------------
MAX_REPLAN_ITERATIONS = int(os.getenv("MAX_REPLAN_ITERATIONS", "3"))

# --- Server ---------------------------------------------------------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
