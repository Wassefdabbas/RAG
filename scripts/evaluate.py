"""
Phase 4: Evaluation — measures retrieval quality with Recall@K and MRR,
using topic keywords (matched against the `topic` column) as ground truth.

Updated to include re-ranking, so this measures the SAME pipeline that's
actually used in production (hybrid search -> re-rank), not just raw
hybrid search.

Run from the project root:
    python -m scripts.evaluate
"""

from src.embeddings.text import embed_text
from src.db.client import supabase
from src.retrieval.reranker import rerank
from src.evaluation.metrics import recall_at_k, reciprocal_rank

CANDIDATE_COUNT = 15
K = 5  # final count after re-ranking, matches the production pipeline

TEST_CASES = [
    ("What do Syrians typically eat during celebrations?", "cuisine"),
    ("What is traditional Syrian clothing like?", "clothing"),
    ("Tell me about Syrian wedding customs.", "social"),
    ("What instruments are used in Syrian music?", "music"),
    ("What crops are grown in rural Syria?", "agriculture"),
    ("What is Aleppo known for historically?", "architecture"),
    ("What languages and dialects are spoken in Syria?", "language"),
    ("What are common Syrian handicrafts?", "crafts"),
    ("What happens during Syrian religious festivals?", "festivals"),
]


def get_relevant_ids(topic_keyword: str) -> set[int]:
    result = (
        supabase.table("documents")
        .select("id, topic")
        .ilike("topic", f"%{topic_keyword}%")
        .execute()
    )
    return {row["id"] for row in result.data}


def retrieve_doc_ids(question: str, k: int) -> list[int]:
    query_embedding = embed_text(question)
    result = supabase.rpc(
        "hybrid_search",
        {"query_text": question, "query_embedding": query_embedding, "match_count": CANDIDATE_COUNT},
    ).execute()

    reranked = rerank(question, result.data, top_k=k)
    return [row["document_id"] for row in reranked]


def main():
    recalls = []
    rr_scores = []

    print(f"Evaluating {len(TEST_CASES)} questions (candidates={CANDIDATE_COUNT}, final K={K})\n")

    for question, topic_keyword in TEST_CASES:
        relevant_ids = get_relevant_ids(topic_keyword)
        if not relevant_ids:
            print(f"⚠️  No document found for topic '{topic_keyword}' — skipping.")
            continue

        retrieved_ids = retrieve_doc_ids(question, k=K)

        recall = recall_at_k(retrieved_ids, relevant_ids, K)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        recalls.append(recall)
        rr_scores.append(rr)

        status = "✅" if recall == 1.0 else "❌"
        print(f"{status} Q: {question}")
        print(f"   expected topic contains '{topic_keyword}' -> ids {relevant_ids}")
        print(f"   retrieved doc_ids (reranked): {retrieved_ids}")
        print(f"   recall@{K}={recall:.1f}  reciprocal_rank={rr:.2f}\n")

    if recalls:
        print("=" * 50)
        print(f"Mean Recall@{K}: {sum(recalls)/len(recalls):.2%}")
        print(f"MRR: {sum(rr_scores)/len(rr_scores):.3f}")


if __name__ == "__main__":
    main()