-- One row per source PDF/document
create table if not exists documents (
    id bigserial primary key,
    file_name text not null,
    topic text,                      -- e.g. 'cuisine', 'clothing', 'music'
    storage_path text,               -- local path or Supabase Storage path
    created_at timestamptz default now()
);