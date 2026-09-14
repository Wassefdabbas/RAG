"""
Data models for the RAG pipeline: what goes INTO an ask (request) and
what comes OUT of one (response), plus the LLM's structured outputs.

Keeping these separate from chain.py means the "shape" of the API is
defined in one obvious place, independent of how the chain is wired.
"""

from pydantic import BaseModel, Field


# ---- LLM structured outputs (what Gemini is forced to return) ----

class RAGResponse(BaseModel):
    """The LLM's own structured answer to a question."""
    answer: str = Field(description="The answer to the question, based only on the given context")
    confidence: str = Field(description="One of: high, medium, low — how well the context supports this answer")
    sources_used: list[int] = Field(description="Which source numbers (e.g. 1, 2) from the context were actually used to answer")


class JudgmentResponse(BaseModel):
    """LLM-as-Judge verdict on a previously generated answer."""
    faithful: bool = Field(description="True if the answer is fully supported by the context, with no invented facts")
    relevant: bool = Field(description="True if the answer actually addresses the question asked")
    reasoning: str = Field(description="One short sentence explaining the judgment")


# ---- Pipeline-level request / response (what ask() takes and returns) ----

class AskRequest(BaseModel):
    """What a caller sends in to ask a question."""
    question: str
    match_count: int = 5


class Source(BaseModel):
    """One retrieved chunk that (may have) informed the answer."""
    document_id: int
    content: str
    score: float
    rerank_score: float | None = None


class Usage(BaseModel):
    """Token/cost accounting for one LLM call."""
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class AskResponse(BaseModel):
    """What ask() returns — and what the API sends back."""
    answer: str
    confidence: str
    sources: list[Source]
    usage: Usage | None = None