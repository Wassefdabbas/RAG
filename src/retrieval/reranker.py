"""
Cross-encoder re-ranking: takes a broader set of candidate chunks from
hybrid search and re-scores them by looking at the query and each
chunk TOGETHER (unlike embeddings, which score them separately) —
more accurate, but slower, so it's only applied to the top N candidates,
not the whole database.
"""

from sentence_transformers import CrossEncoder

from src.core.logger import get_logger

logger = get_logger(__name__)

# English-only, consistent with the embedding model chosen for this
# learning phase. Swap for a multilingual cross-encoder later if needed.
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """candidates must be dicts with a 'content' key (as returned by
    hybrid_search). Returns the top_k candidates, re-scored and re-sorted,
    each with an added 'rerank_score' field."""

    if not candidates:
        return []

    pairs = [(query, c["content"]) for c in candidates]
    scores = _model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(candidates)} candidates -> top {top_k} "
        f"(best score={reranked[0]['rerank_score']:.3f})"
    )

    return reranked[:top_k]