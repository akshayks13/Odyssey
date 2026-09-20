"""Weighted cosine match between user preferences and place category scores."""
from __future__ import annotations

import math

from langchain_core.tools import tool

_CATEGORY_KEYS = ["nature", "adventure", "food", "nightlife", "relaxation", "culture", "shopping"]


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = _CATEGORY_KEYS
    va = [a.get(k, 0.0) for k in keys]
    vb = [b.get(k, 0.0) for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@tool(parse_docstring=True)
def score_preference_match(preferences: dict[str, float], category_scores: dict[str, float]) -> dict:
    """Score how well a destination/activity matches user preferences.

    Args:
        preferences: User's preference weights, e.g. {"nature": 0.9, "adventure": 0.8, ...}.
        category_scores: Destination/activity's category scores on the same scale.

    Returns:
        dict with score in [0, 1] (cosine similarity).
    """
    score = cosine_similarity(preferences, category_scores)
    return {"score": round(max(0.0, min(1.0, score)), 4)}
