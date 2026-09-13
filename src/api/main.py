"""
FastAPI app exposing the RAG pipeline as an HTTP API, protected by an
API key and basic input validation.

Run from the project root:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.core.logger import get_logger
from src.rag.chain import ask as rag_ask
from src.api.security import verify_api_key
from src.api.input_guard import validate_question, InputValidationError

logger = get_logger(__name__)

app = FastAPI(title="MMCQA-RAG API", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[dict]
    usage: dict | None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_endpoint(request: AskRequest):
    try:
        question = validate_question(request.question)
    except InputValidationError as e:
        logger.warning(f"Rejected input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Received question: {question}")

    try:
        result = rag_ask(question)
    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    return AskResponse(**result)