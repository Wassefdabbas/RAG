"""
Tests for the in-memory RAG answer cache.
"""

import time
import pytest

from src.rag import cache as cache_module


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the module-level cache dicts before/after each test so
    tests don't leak state into each other."""
    cache_module._cache.clear()
    cache_module._timestamps.clear()
    yield
    cache_module._cache.clear()
    cache_module._timestamps.clear()


class TestCache:
    def test_miss_when_nothing_cached(self):
        assert cache_module.get_cached("some question", 5) is None

    def test_hit_after_set(self):
        result = {"answer": "Damascus", "confidence": "high"}
        cache_module.set_cached("What is the capital?", 5, result)
        assert cache_module.get_cached("What is the capital?", 5) == result

    def test_case_and_whitespace_insensitive_key(self):
        result = {"answer": "Damascus"}
        cache_module.set_cached("What is the capital?", 5, result)
        assert cache_module.get_cached("  WHAT IS THE CAPITAL?  ", 5) == result

    def test_different_match_count_is_a_different_key(self):
        cache_module.set_cached("question", 5, {"answer": "A"})
        assert cache_module.get_cached("question", 3) is None

    def test_expired_entry_returns_none(self, monkeypatch):
        cache_module.set_cached("question", 5, {"answer": "A"})
        # simulate time passing beyond the TTL
        key = cache_module._cache_key("question", 5)
        cache_module._timestamps[key] = time.time() - (cache_module.CACHE_TTL_SECONDS + 1)
        assert cache_module.get_cached("question", 5) is None