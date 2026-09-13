"""
Full RAG pipeline with structured output, caching, and re-ranking:
question -> hybrid retrieval (broad) -> re-rank (precise) -> prompt
-> LLM (JSON schema) -> structured answer + sources.
"""

from src.core.logger import get_logger
from src.embeddings.text import embed_text
from src.db.client import supabase
from src.retrieval.reranker import rerank
from src.rag.gemini_client import generate_structured_answer
from src.rag.schemas import RAGResponse
from src.rag.cache import get_cached, set_cached

logger = get_logger(__name__)

# Fetch a wider net from hybrid search, then re-rank down to a smaller,
# more precise final set before it goes to the LLM.
CANDIDATE_COUNT = 15
FINAL_COUNT = 5

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Syrian culture.
Use ONLY the context below to answer the question. If the context doesn't
contain enough information to answer, say so honestly in the answer field
and set confidence to "low".

Context:
{context}

Question: {question}"""


def retrieve(question: str, candidate_count: int = CANDIDATE_COUNT, final_count: int = FINAL_COUNT) -> list[dict]:
    query_embedding = embed_text(question)

    result = supabase.rpc(
        "hybrid_search",
        {
            "query_text": question,
            "query_embedding": query_embedding,
            "match_count": candidate_count,
        },
    ).execute()

    candidates = result.data
    return rerank(question, candidates, top_k=final_count)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source {i+1}] {chunk['content']}" for i, chunk in enumerate(chunks)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def ask(question: str, use_cache: bool = True) -> dict:
    if use_cache:
        cached = get_cached(question, FINAL_COUNT)
        if cached is not None:
            return cached

    chunks = retrieve(question)

    if not chunks:
        logger.info("No chunks retrieved — skipping LLM call.")
        result = {
            "answer": "I couldn't find any relevant information to answer that.",
            "confidence": "low",
            "sources": [],
            "usage": None,
        }
        if use_cache:
            set_cached(question, FINAL_COUNT, result)
        return result

    prompt = build_prompt(question, chunks)
    llm_result = generate_structured_answer(prompt, schema=RAGResponse)
    parsed: RAGResponse = llm_result["parsed"]

    result = {
        "answer": parsed.answer,
        "confidence": parsed.confidence,
        "sources": [
            {
                "document_id": chunks[i - 1]["document_id"],
                "content": chunks[i - 1]["content"][:200],
                "score": chunks[i - 1]["score"],
                "rerank_score": chunks[i - 1]["rerank_score"],
            }
            for i in parsed.sources_used
            if 1 <= i <= len(chunks)
        ],
        "usage": {
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "estimated_cost_usd": llm_result["estimated_cost_usd"],
        },
    }

    if use_cache:
        set_cached(question, FINAL_COUNT, result)

    return result