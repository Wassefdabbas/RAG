from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents

file_path = "data/raw/docs/CV.pdf"

documents = load_pdf(file_path)
documents = clean_documents(documents)


print(f"Pages loaded: {len(documents)}")

chunks = chunk_documents(documents)

print(f"Chunks created: {len(chunks)}")

print("\n--- First chunk ---")
print(chunks[0].page_content)

print("\n--- Metadata ---")
print(chunks[0].metadata)