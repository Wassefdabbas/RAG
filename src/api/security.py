"""
Simple API key authentication for the RAG API.

This is the "real job" pattern for internal/small APIs: a shared secret
key passed in a header, checked on every request. Bigger production
systems often move to OAuth2/JWT with per-user identity, but API keys
are still extremely common for service-to-service or early-stage APIs.
"""

from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

from src.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")