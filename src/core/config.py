"""
Central configuration for the project.
"""

import os
import secrets
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Check your .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_key: str
    gemini_api_key: str

    # API security — a shared secret clients must send in the X-API-Key header.
    # Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    api_key: str = ""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chunk_size: int = 500
    chunk_overlap: int = 50

    llm_model_name: str = "gemini-3.5-flash-lite"
    llm_rpm_limit: int = 15
    llm_tpm_limit: int = 250_000
    llm_rpd_limit: int = 500

    llm_input_price_per_1m: float = 0.30
    llm_output_price_per_1m: float = 2.50


def load_settings() -> Settings:
    return Settings(
        supabase_url=_require("SUPABASE_URL"),
        supabase_key=_require("SUPABASE_KEY"),
        gemini_api_key=_require("GEMINI_API_KEY"),
        api_key=_require("API_KEY"),
    )


settings = load_settings()