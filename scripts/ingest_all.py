"""
Full ingestion pipeline: reads every PDF in data/raw/docs/,
loads -> cleans -> chunks -> embeds -> stores in Supabase.

Run from the project root:
    python -m scripts.ingest_all
"""

from pathlib import Path
import re

from src.ingestion.loader import load_pdf
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.embeddings.text import embed_texts
from src.db.client import supabase

DOCS_DIR = Path("data/raw/docs")


def ingest_file(file_path: Path) -> None:
    print(f"\n--- Processing {file_path.name} ---")

    # 1. Load
    docs = load_pdf(str(file_path))

    # 2. Clean
    docs = clean_documents(docs)
    if not docs:
        print(f"  Skipped: no content left after cleaning.")
        return

    # 3. Chunk
    chunks = chunk_documents(docs)
    print(f"  Loaded {len(docs)} page(s), created {len(chunks)} chunk(s).")

    # Insert the document row first, so we have a document_id to link chunks to.
    # topic is derived from the filename (e.g. "cuisine.pdf" -> "cuisine");
    # adjust manually if your filenames don't map cleanly to a topic.
    topic = re.sub(r"^\d+[-_]\s*", "", file_path.stem).replace("_", " ").replace("-", " ")

    doc_result = (
        supabase.table("documents")
        .insert({
            "file_name": file_path.name,
            "topic": topic,
            "storage_path": str(file_path),
        })
        .execute()
    )
    document_id = doc_result.data[0]["id"]

    # 4. Embed all chunks in one batch (faster than one-by-one)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embed_texts(texts)

    # 5. Store each chunk with its embedding
    rows = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        rows.append({
            "document_id": document_id,
            "chunk_index": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": vector,
        })

    supabase.table("document_chunks").insert(rows).execute()
    print(f"  Stored {len(rows)} chunk(s) under document_id={document_id}.")


def main() -> None:
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCS_DIR}/")
        return

    print(f"Found {len(pdf_files)} PDF(s) to ingest.")
    for file_path in pdf_files:
        ingest_file(file_path)

    print("\nDone.")


if __name__ == "__main__":
    main()