"""
Gemini LLM client with rate limiting, cost tracking, and logging.
"""

from google import genai
from pydantic import BaseModel

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.rate_limiter import rate_limiter

logger = get_logger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)


def _rough_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * settings.llm_input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * settings.llm_output_price_per_1m
    return input_cost + output_cost


def _log_and_track_usage(response) -> dict:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count

    rate_limiter.record_usage(total_tokens)
    cost = _estimate_cost(input_tokens, output_tokens)

    logger.info(
        f"Response received | input_tokens={input_tokens} "
        f"output_tokens={output_tokens} est_cost=${cost:.6f}"
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }


def generate_answer(prompt: str) -> dict:
    """Plain text answer (no structured output)."""
    estimated_tokens = _rough_token_estimate(prompt)
    rate_limiter.wait_if_needed(estimated_tokens)

    logger.info(f"Calling {settings.llm_model_name} (~{estimated_tokens} est. input tokens)")

    response = _client.models.generate_content(
        model=settings.llm_model_name,
        contents=prompt,
    )

    usage = _log_and_track_usage(response)
    return {"text": response.text, **usage}


def generate_structured_answer(prompt: str, schema: type[BaseModel]) -> dict:
    """Same as generate_answer, but forces the model to return JSON matching
    the given Pydantic schema, parsed automatically into that schema."""

    estimated_tokens = _rough_token_estimate(prompt)
    rate_limiter.wait_if_needed(estimated_tokens)

    logger.info(
        f"Calling {settings.llm_model_name} for structured output "
        f"(~{estimated_tokens} est. input tokens, schema={schema.__name__})"
    )

    response = _client.models.generate_content(
        model=settings.llm_model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    usage = _log_and_track_usage(response)
    parsed = schema.model_validate_json(response.text)

    return {"parsed": parsed, **usage}