-- Similarity search function: given a query embedding, return the top matching chunks
create or replace function match_chunks (
    query_embedding vector(384),
    match_threshold float default 0.5,
    match_count int default 5
)
returns table (
    id bigint,
    document_id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        document_chunks.id,
        document_chunks.document_id,
        document_chunks.content,
        document_chunks.metadata,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from document_chunks
    where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
    order by document_chunks.embedding <=> query_embedding
    limit match_count;
$$;