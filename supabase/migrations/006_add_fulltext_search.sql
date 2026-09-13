-- Add a generated tsvector column for keyword/full-text search,
-- kept in sync automatically whenever `content` changes.
alter table document_chunks
    add column if not exists fts tsvector
    generated always as (to_tsvector('english', content)) stored;

-- GIN index makes full-text search fast
create index if not exists document_chunks_fts_idx
    on document_chunks
    using gin (fts);