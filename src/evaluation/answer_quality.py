"""
Answer quality evaluation using LLM-as-Judge: a separate call asks the
model to judge whether a generated answer is faithful to its context
and relevant to the question — catching hallucination and off-topic
answers that Recall@K / MRR can't measure (those only check retrieval).
"""

from src.rag.gemini_client import generate_structured_answer
from src.rag.schemas import JudgmentResponse
from src.rag.prompts import JUDGE_PROMPT


def judge_answer(question: str, context: str, answer: str) -> JudgmentResponse:
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    result = generate_structured_answer(prompt, schema=JudgmentResponse)
    return result["parsed"]