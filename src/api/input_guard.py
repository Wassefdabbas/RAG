"""
Basic input validation / prompt-injection mitigation.

Note: there is no 100% reliable defense against prompt injection with
current LLMs — this is risk REDUCTION, not elimination. Real production
systems combine several of these layers, plus monitoring, and accept
some residual risk.
"""

MAX_QUESTION_LENGTH = 500

# Crude keyword-based filter — catches obvious attempts, not sophisticated ones.
# Real systems often add a dedicated classifier model for this instead.
SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard the context",
    "you are now",
    "system prompt",
    "reveal your instructions",
    "act as if",
]


class InputValidationError(Exception):
    pass


def validate_question(question: str) -> str:
    question = question.strip()

    if not question:
        raise InputValidationError("Question cannot be empty.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise InputValidationError(
            f"Question too long ({len(question)} chars, max {MAX_QUESTION_LENGTH})."
        )

    lowered = question.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in lowered:
            raise InputValidationError(
                "Question contains a disallowed instruction pattern."
            )

    return question