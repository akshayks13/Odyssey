"""Cosine similarity between the traveller's preference vector and a place's category profile."""
from __future__ import annotations

import math

CATEGORY_KEYS = ["nature", "adventure", "food", "nightlife", "relaxation", "culture", "shopping"]


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    va = [a.get(k, 0.0) for k in CATEGORY_KEYS]
    vb = [b.get(k, 0.0) for k in CATEGORY_KEYS]
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(y * y for y in vb))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_preference_match(preferences: dict[str, float], category_scores: dict[str, float]) -> float:
    """Match in [0, 1]."""
    return round(max(0.0, min(1.0, cosine_similarity(preferences, category_scores))), 4)


def sight_preference(preferences: dict[str, float], category: str) -> float:
    """How much the traveller wants a sight of this category: their weight for it, relative to their top weight."""
    top = max(preferences.values()) if preferences else 1.0
    return round(preferences.get(category, 0.3) / top, 4) if top > 0 else 0.5
