"""

The RAG pipeline, built as a LangChain chain (LCEL): retriever ->
prompt -> structured LLM call, with caching, rate limiting, cost
tracking, and cross-encoder re-ranking built in.

Note on design: `with_structured_output` normally requires a full
ChatModel (not our simpler rate-limited LLM class). Instead of rebuilding
the LLM as a ChatModel, we use RunnableLambda to plug our existing
generate_structured_answer() call into the chain — this keeps rate
limiting, cost tracking, and Gemini's native JSON schema mode intact,
while the chain is still a real LCEL composition
(retriever | format | prompt | structured-call).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from src.core.logger import get_logger
from src.retrieval.langchain_retriever import HybridSupabaseRetriever
from src.rag.gemini_client import generate_structured_answer
from src.rag.schemas import RAGResponse
from src.rag.cache import get_cached, set_cached

logger = get_logger(__name__)

MATCH_COUNT = 5
retriever = HybridSupabaseRetriever(match_count=MATCH_COUNT)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about Syrian culture.
Use ONLY the context below to answer the question. If the context doesn't
contain enough information to answer, say so honestly in the answer field
and set confidence to "low".

Context:
{context}

Question: {question}"""
)


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
    | prompt
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
        result = {
            "answer": "I couldn't find any relevant information to answer that.",
            "confidence": "low",
            "sources": [],
            "usage": None,
        }
        if use_cache:
            set_cached(question, MATCH_COUNT, result)
        return result

    llm_result = rag_chain.invoke(question)
    parsed: RAGResponse = llm_result["parsed"]

    result = {
        "answer": parsed.answer,
        "confidence": parsed.confidence,
        "sources": [
            {
                "document_id": docs[i - 1].metadata["document_id"],
                "content": docs[i - 1].page_content[:200],
                "score": docs[i - 1].metadata["score"],
            }
            for i in parsed.sources_used
            if 1 <= i <= len(docs)
        ],
        "usage": {
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "estimated_cost_usd": llm_result["estimated_cost_usd"],
        },
    }

    if use_cache:
        set_cached(question, MATCH_COUNT, result)

    return result