"""
All prompt templates live here — separate from the pipeline logic, so
you can tune wording, add few-shot examples, or A/B test phrasing
without touching src/rag/chain.py.
"""

from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about Syrian culture.
Use ONLY the context below to answer the question. If the context doesn't
contain enough information to answer, say so honestly in the answer field
and set confidence to "low".

Context:
{context}

Question: {question}"""
)

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict evaluator. Judge the ANSWER below
against the CONTEXT it was supposed to be based on.

- faithful: true only if every claim in the answer is supported by the context.
  If the answer adds any fact not present in the context, faithful must be false.
- relevant: true only if the answer actually addresses the question asked.

Context:
{context}

Question: {question}

Answer to judge:
{answer}"""
)