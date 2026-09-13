"""
Tests for the ingestion pipeline: cleaning and chunking.
These don't need Supabase or any network access — pure logic tests.
"""

import pytest
from langchain_core.documents import Document

from src.ingestion.cleaner import clean_text, clean_documents
from src.ingestion.chunker import chunk_documents


class TestCleanText:
    def test_removes_hyphenated_line_breaks(self):
        raw = "This is a docu-\nment about Syria."
        result = clean_text(raw)
        assert "docu-\n" not in result
        assert "document" in result

    def test_collapses_multiple_newlines(self):
        raw = "Paragraph one.\n\n\n\nParagraph two."
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_replaces_single_newlines_with_space(self):
        raw = "This sentence\nwraps across a line."
        result = clean_text(raw)
        assert "\n" not in result or result.count("\n") == 0

    def test_strips_leading_trailing_whitespace(self):
        raw = "   some text   "
        assert clean_text(raw) == "some text"

    def test_empty_string_stays_empty(self):
        assert clean_text("") == ""


class TestCleanDocuments:
    def test_drops_documents_that_become_empty(self):
        docs = [
            Document(page_content="   ", metadata={"source": "blank.pdf"}),
            Document(page_content="Real content here.", metadata={"source": "real.pdf"}),
        ]
        result = clean_documents(docs)
        assert len(result) == 1
        assert result[0].metadata["source"] == "real.pdf"

    def test_preserves_metadata(self):
        docs = [Document(page_content="Some text.", metadata={"source": "x.pdf", "page": 1})]
        result = clean_documents(docs)
        assert result[0].metadata == {"source": "x.pdf", "page": 1}


class TestChunkDocuments:
    def test_splits_long_document_into_multiple_chunks(self):
        long_text = "Sentence about Syria. " * 100  # long enough to force multiple chunks
        docs = [Document(page_content=long_text, metadata={"source": "long.pdf"})]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_short_document_stays_one_chunk(self):
        docs = [Document(page_content="Short text.", metadata={"source": "short.pdf"})]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_chunks_preserve_source_metadata(self):
        docs = [Document(page_content="Text. " * 50, metadata={"source": "test.pdf"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert all(c.metadata["source"] == "test.pdf" for c in chunks)