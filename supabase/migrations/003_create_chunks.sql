-- One row per chunk of a document, with its embedding
create table if not exists document_chunks (
    id bigserial primary key,
    document_id bigint references documents(id) on delete cascade,
    chunk_index int not null,
    content text not null,
    metadata jsonb default '{}'::jsonb,
    embedding vector(384),           -- matches all-MiniLM-L6-v2
    created_at timestamptz default now()
);