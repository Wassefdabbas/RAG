"""
Phase 4: Answer quality evaluation — runs the full RAG pipeline (LangChain
chain) on each test question, then judges the generated answer for
faithfulness and relevance using LLM-as-Judge.

Note: this makes 2 LLM calls per question (1 to answer, 1 to judge),
so it uses more of your free-tier quota than scripts/evaluate.py.

Run from the project root:
    python -m scripts.evaluate_answers
"""

from src.rag.chain import ask, retriever
from src.evaluation.answer_quality import judge_answer

TEST_QUESTIONS = [
    "What do Syrians typically eat during celebrations?",
    "What is traditional Syrian clothing like?",
    "Tell me about Syrian wedding customs.",
    "What instruments are used in Syrian music?",
    "What crops are grown in rural Syria?",
]


def main():
    faithful_count = 0
    relevant_count = 0
    total = 0

    for question in TEST_QUESTIONS:
        docs = retriever.invoke(question)
        if not docs:
            print(f"⚠️  No chunks for: {question}")
            continue

        result = ask(question)
        context = "\n\n".join(d.page_content for d in docs)

        judgment = judge_answer(question, context, result["answer"])

        total += 1
        faithful_count += int(judgment.faithful)
        relevant_count += int(judgment.relevant)

        status = "✅" if judgment.faithful and judgment.relevant else "❌"
        print(f"\n{status} Q: {question}")
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   faithful={judgment.faithful}  relevant={judgment.relevant}")
        print(f"   reasoning: {judgment.reasoning}")

    if total:
        print("\n" + "=" * 50)
        print(f"Faithfulness rate: {faithful_count/total:.1%}")
        print(f"Relevance rate:    {relevant_count/total:.1%}")


if __name__ == "__main__":
    main()