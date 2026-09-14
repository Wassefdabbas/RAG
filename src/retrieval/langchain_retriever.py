"""
LangChain-compatible retriever wrapping our Supabase hybrid_search RPC,
with cross-encoder re-ranking applied before returning results.
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from src.embeddings.text import embed_text
from src.db.client import supabase
from src.retrieval.reranker import rerank


class HybridSupabaseRetriever(BaseRetriever):
    candidate_count: int = Field(default=15)
    final_count: int = Field(default=5)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = embed_text(query)

        result = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": query_embedding,
                "match_count": self.candidate_count,
            },
        ).execute()

        reranked = rerank(query, result.data, top_k=self.final_count)

        return [
            Document(
                page_content=row["content"],
                metadata={
                    "document_id": row["document_id"],
                    "score": row["score"],
                    "rerank_score": row["rerank_score"],
                    "chunk_id": row["id"],
                },
            )
            for row in reranked
        ]