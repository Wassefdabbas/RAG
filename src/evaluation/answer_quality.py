"""
Answer quality evaluation using LLM-as-Judge: a separate call asks the
model to judge whether a generated answer is faithful to its context
and relevant to the question — catching hallucination and off-topic
answers that Recall@K / MRR can't measure (those only check retrieval).
"""

from src.rag.gemini_client import generate_structured_answer
from src.rag.schemas import JudgmentResponse

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator. Judge the ANSWER below
against the CONTEXT it was supposed to be based on.

- faithful: true only if every claim in the answer is supported by the context.
  If the answer adds any fact not present in the context, faithful must be false.
- relevant: true only if the answer actually addresses the question asked.

Context:
{context}

Question: {question}

Answer to judge:
{answer}"""


def judge_answer(question: str, context: str, answer: str) -> JudgmentResponse:
    prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    result = generate_structured_answer(prompt, schema=JudgmentResponse)
    return result["parsed"]