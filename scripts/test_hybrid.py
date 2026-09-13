"""
Compare semantic-only search vs hybrid (semantic + keyword) search
on the same questions.

Run from the project root:
    python -m scripts.test_hybrid
"""

from src.embeddings.text import embed_text
from src.db.client import supabase


def hybrid_search(query: str, match_count: int = 5):
    query_embedding = embed_text(query)

    result = supabase.rpc(
        "hybrid_search",
        {
            "query_text": query,
            "query_embedding": query_embedding,
            "match_count": match_count,
        },
    ).execute()

    return result.data


def main():
    questions = [
        "What do Syrians typically eat during celebrations?",
        "What is traditional Syrian clothing like?",
        "Tell me about Syrian wedding customs.",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        matches = hybrid_search(q)
        if not matches:
            print("  No matches found.")


        MAX_RRF_SCORE = 2 * (1 / 50)
        for m in matches:
            percentage = (m['score'] / MAX_RRF_SCORE) * 100
            print(f"\n  [score={percentage:.1f}%] (doc_id={m['document_id']})")
            print(f"  {m['content'][:200]}...")

if __name__ == "__main__":
    main()