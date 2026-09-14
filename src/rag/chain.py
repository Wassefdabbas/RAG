"""
The RAG pipeline, built as a LangChain chain (LCEL): retriever ->
prompt -> structured LLM call, with caching, rate limiting, cost
tracking, and cross-encoder re-ranking built in.

Prompts live in src/rag/prompts.py. Request/response shapes and LLM
structured-output schemas live in src/rag/schemas.py.
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from src.core.logger import get_logger
from src.retrieval.langchain_retriever import HybridSupabaseRetriever
from src.rag.gemini_client import generate_structured_answer
from src.rag.prompts import ANSWER_PROMPT
from src.rag.schemas import RAGResponse, Source, Usage, AskResponse
from src.rag.cache import get_cached, set_cached

logger = get_logger(__name__)

MATCH_COUNT = 5
retriever = HybridSupabaseRetriever(match_count=MATCH_COUNT)


def format_docs_with_sources(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)
    )


def _generate_structured(prompt_value) -> dict:
    """RunnableLambda step: takes the rendered prompt and calls Gemini
    with a JSON schema, returning the parsed response + usage info."""
    prompt_text = prompt_value.to_string()
    return generate_structured_answer(prompt_text, schema=RAGResponse)


rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | ANSWER_PROMPT
    | RunnableLambda(_generate_structured)
)


def ask(question: str, use_cache: bool = True) -> dict:
    if use_cache:
        cached = get_cached(question, MATCH_COUNT)
        if cached is not None:
            return cached

    docs = retriever.invoke(question)

    if not docs:
        logger.info("No chunks retrieved — skipping LLM call.")
        result = AskResponse(
            answer="I couldn't find any relevant information to answer that.",
            confidence="low",
            sources=[],
            usage=None,
        ).model_dump()
        if use_cache:
            set_cached(question, MATCH_COUNT, result)
        return result

    llm_result = rag_chain.invoke(question)
    parsed: RAGResponse = llm_result["parsed"]

    result = AskResponse(
        answer=parsed.answer,
        confidence=parsed.confidence,
        sources=[
            Source(
                document_id=docs[i - 1].metadata["document_id"],
                content=docs[i - 1].page_content[:200],
                score=docs[i - 1].metadata["score"],
                rerank_score=docs[i - 1].metadata.get("rerank_score"),
            )
            for i in parsed.sources_used
            if 1 <= i <= len(docs)
        ],
        usage=Usage(
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            estimated_cost_usd=llm_result["estimated_cost_usd"],
        ),
    ).model_dump()

    if use_cache:
        set_cached(question, MATCH_COUNT, result)

    return result