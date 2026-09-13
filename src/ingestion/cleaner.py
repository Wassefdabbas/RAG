"""
Basic text cleaning for documents loaded from PDFs, before chunking.
Keep this intentionally simple and easy to extend — add rules here
as you notice more issues in your actual documents.
"""

import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Apply basic cleanup rules to raw extracted text."""

    # Fix words broken across a line break with a hyphen, e.g. "docu-\nment" -> "document"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Collapse multiple newlines into a single paragraph break
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace single newlines (mid-sentence line wraps) with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse repeated whitespace/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace on the whole text
    text = text.strip()

    return text


def clean_documents(documents: list[Document]) -> list[Document]:
    """Apply clean_text to a list of langchain Documents, preserving metadata."""
    cleaned = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        # skip pages that end up empty after cleaning (e.g. blank pages)
        if cleaned_content:
            cleaned.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    return cleaned