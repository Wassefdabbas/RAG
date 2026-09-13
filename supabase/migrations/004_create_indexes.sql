-- Speeds up similarity search once you have more than a handful of rows.
-- Safe to run even with a small dataset now.
create index if not exists document_chunks_embedding_idx
    on document_chunks
    using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_document_id_idx
    on document_chunks (document_id);