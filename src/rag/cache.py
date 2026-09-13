"""
Simple in-memory cache for RAG answers.

Keyed by the normalized question + match_count, with a time-to-live (TTL)
so answers don't go stale forever if the underlying documents change.

This is process-local (resets if you restart the server) — good enough
for a learning project / single-instance API. A real production setup
with multiple server instances would need a shared cache (e.g. Redis)
instead of this in-memory dict.
"""

import time
import hashlib

from src.core.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 3600  # 1 hour

_cache: dict[str, dict] = {}
_timestamps: dict[str, float] = {}


def _cache_key(question: str, match_count: int) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(f"{normalized}:{match_count}".encode()).hexdigest()


def get_cached(question: str, match_count: int) -> dict | None:
    key = _cache_key(question, match_count)
    if key not in _cache:
        return None

    age = time.time() - _timestamps[key]
    if age > CACHE_TTL_SECONDS:
        del _cache[key]
        del _timestamps[key]
        return None

    logger.info(f"Cache hit ({age:.0f}s old): {question!r}")
    return _cache[key]


def set_cached(question: str, match_count: int, result: dict) -> None:
    key = _cache_key(question, match_count)
    _cache[key] = result
    _timestamps[key] = time.time()


def cache_stats() -> dict:
    return {"entries": len(_cache)}