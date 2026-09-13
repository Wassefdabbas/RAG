-- Hybrid search: combines semantic (vector) search and keyword
-- (full-text) search using Reciprocal Rank Fusion (RRF).
--
-- Note: we turn the query into an OR-based tsquery (word1 OR word2 OR ...)
-- instead of the default AND. A natural-language question rarely has all
-- its words appear together in one chunk, so AND (the websearch_to_tsquery
-- default) returns zero full-text matches for most real questions.
create or replace function hybrid_search(
    query_text text,
    query_embedding vector(384),
    match_count int default 5,
    full_text_weight float default 1.0,
    semantic_weight float default 1.0,
    rrf_k int default 50
)
returns table (
    id bigint,
    document_id bigint,
    content text,
    metadata jsonb,
    score float
)
language sql
as $$
with full_text as (
    select
        document_chunks.id,
        row_number() over (
            order by ts_rank_cd(
                document_chunks.fts,
                websearch_to_tsquery('english', regexp_replace(trim(query_text), '\s+', ' or ', 'g'))
            ) desc
        ) as rank_ix
    from document_chunks
    where document_chunks.fts @@ websearch_to_tsquery('english', regexp_replace(trim(query_text), '\s+', ' or ', 'g'))
    limit least(match_count, 30) * 2
),
semantic as (
    select
        document_chunks.id,
        row_number() over (
            order by document_chunks.embedding <=> query_embedding
        ) as rank_ix
    from document_chunks
    limit least(match_count, 30) * 2
)
select
    document_chunks.id,
    document_chunks.document_id,
    document_chunks.content,
    document_chunks.metadata,
    (
        coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
        coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
    ) as score
from full_text
full outer join semantic on full_text.id = semantic.id
join document_chunks on document_chunks.id = coalesce(full_text.id, semantic.id)
order by score desc
limit match_count;
$$;