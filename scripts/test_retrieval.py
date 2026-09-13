"""
Test retrieval WITHOUT an LLM — just embed a question and see which
chunks come back. This is how you check if retrieval itself is good,
before adding any LLM on top of it.

Run from the project root:
    python -m scripts.test_retrieval
"""

from src.embeddings.text import embed_text
from src.db.client import supabase


def search(query: str, match_count: int = 5, match_threshold: float = 0.3):
    query_embedding = embed_text(query)

    result = supabase.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
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
        matches = search(q)
        if not matches:
            print("  No matches found (try lowering match_threshold).")
        for m in matches:
            print(f"\n  [similarity={m['similarity']:.3f}] (doc_id={m['document_id']})")
            print(f"  {m['content'][:200]}...")


if __name__ == "__main__":
    main()