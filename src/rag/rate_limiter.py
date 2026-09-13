"""
Simple in-memory rate limiter for the free-tier LLM limits:
requests/minute, tokens/minute, and requests/day.

This is intentionally simple (in-process, resets if you restart the
script) — good enough for a learning project run from one machine.
For a real multi-process/production setup, this state would need to
live somewhere shared (e.g. Redis), not in Python memory.
"""

import time
from collections import deque

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    def __init__(self):
        self._minute_requests: deque[float] = deque()
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._day_requests: deque[float] = deque()

    def _prune(self, now: float):
        while self._minute_requests and now - self._minute_requests[0] > 60:
            self._minute_requests.popleft()
        while self._minute_tokens and now - self._minute_tokens[0][0] > 60:
            self._minute_tokens.popleft()
        while self._day_requests and now - self._day_requests[0] > 86400:
            self._day_requests.popleft()

    def wait_if_needed(self, estimated_tokens: int = 0):
        """Block (sleep) if we'd exceed a limit, then record this call."""
        now = time.time()
        self._prune(now)

        # Requests per day — hard stop, no point waiting a whole day
        if len(self._day_requests) >= settings.llm_rpd_limit:
            raise RuntimeError(
                f"Daily request limit reached ({settings.llm_rpd_limit}/day). "
                f"Try again tomorrow or upgrade your plan."
            )

        # Requests per minute — wait until the oldest request ages out
        if len(self._minute_requests) >= settings.llm_rpm_limit:
            sleep_for = 60 - (now - self._minute_requests[0]) + 0.1
            logger.info(f"RPM limit hit, sleeping {sleep_for:.1f}s")
            time.sleep(max(sleep_for, 0))
            now = time.time()
            self._prune(now)

        # Tokens per minute — wait until enough old token usage ages out
        current_tpm = sum(t for _, t in self._minute_tokens)
        if current_tpm + estimated_tokens > settings.llm_tpm_limit:
            sleep_for = 60 - (now - self._minute_tokens[0][0]) + 0.1
            logger.info(f"TPM limit would be exceeded, sleeping {sleep_for:.1f}s")
            time.sleep(max(sleep_for, 0))
            now = time.time()
            self._prune(now)

        self._minute_requests.append(now)
        self._day_requests.append(now)

    def record_usage(self, tokens_used: int):
        self._minute_tokens.append((time.time(), tokens_used))


# Shared instance — import this wherever you call the LLM
rate_limiter = RateLimiter()