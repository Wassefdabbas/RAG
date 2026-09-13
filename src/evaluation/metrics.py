"""
Retrieval evaluation metrics.
"""


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """1.0 if any relevant document appears in the top k retrieved, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """1/rank of the first relevant document found, or 0.0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0