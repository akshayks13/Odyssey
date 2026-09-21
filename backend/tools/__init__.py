"""Tool layer: LangChain tools used by the agents, plus a small helper for parallel lookups."""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor


def pmap(fn, items, workers: int = 6) -> list:
    """Run `fn` over `items` in threads, keeping order. A failing item gives None instead of raising."""
    items = list(items)

    def run(item):
        try:
            return fn(item)
        except Exception:  # noqa: BLE001
            return None

    if len(items) <= 1:
        return [run(i) for i in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return [f.result() for f in [pool.submit(contextvars.copy_context().run, run, i) for i in items]]
