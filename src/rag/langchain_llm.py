"""
LangChain-compatible LLM wrapper around our rate-limited, cost-tracked
Gemini client — so switching to LangChain chains doesn't lose the
rate limiting / cost tracking we already built.
"""

from typing import Any, Optional
from langchain_core.language_models.llms import LLM

from src.rag.gemini_client import generate_answer
from src.core.logger import get_logger

logger = get_logger(__name__)


class RateLimitedGeminiLLM(LLM):
    """Drop-in LangChain LLM that internally uses our existing
    rate limiter + cost tracker instead of calling the API directly."""

    last_usage: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "rate-limited-gemini"

    def _call(self, prompt: str, stop: Optional[list[str]] = None, **kwargs: Any) -> str:
        result = generate_answer(prompt)
        self.last_usage = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "estimated_cost_usd": result["estimated_cost_usd"],
        }
        return result["text"]