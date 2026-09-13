"""Structured response schemas for the RAG pipeline."""

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question, based only on the given context")
    confidence: str = Field(description="One of: high, medium, low — how well the context supports this answer")
    sources_used: list[int] = Field(description="Which source numbers (e.g. 1, 2) from the context were actually used to answer")


class JudgmentResponse(BaseModel):
    faithful: bool = Field(description="True if the answer is fully supported by the context, with no invented facts")
    relevant: bool = Field(description="True if the answer actually addresses the question asked")
    reasoning: str = Field(description="One short sentence explaining the judgment")