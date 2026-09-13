# MMCQA-RAG

A Retrieval-Augmented Generation (RAG) system built as a learning project

The corpus is 10 short documents on Syrian culture (cuisine, clothing, music, architecture, crafts, social customs, agriculture, festivals, language, and a general overview).

## Architecture

```
PDF documents
     |
     v
  Load (PyPDFLoader)
     |
     v
  Clean (fix line breaks, whitespace)
     |
     v
  Chunk (RecursiveCharacterTextSplitter)
     |
     v
  Embed (sentence-transformers/all-MiniLM-L6-v2)
     |
     v
  Store (Supabase / pgvector)
     |
     v
  Hybrid Search (semantic + full-text, combined via RRF)
     |
     v
  Re-rank (cross-encoder/ms-marco-MiniLM-L-6-v2)
     |
     v
  LLM (Gemini 3.5 Flash Lite, structured JSON output)
     |
     v
  Answer + confidence + sources
```

Exposed as a FastAPI service, protected by an API key, with caching,
rate limiting, cost tracking, LangSmith tracing, and an evaluation suite.

## Tech stack

- **Language / tooling:** Python 3.11, `uv`
- **Orchestration:** LangChain (LCEL)
- **Vector store:** Supabase (PostgreSQL + pgvector)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, English)
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Gemini 3.5 Flash Lite (free tier)
- **API:** FastAPI + Uvicorn
- **Observability:** LangSmith
- **Testing:** pytest (36 tests, no network calls)

## Project structure

```
MMCQA-RAG/
├── data/raw/docs/          # source PDFs (not tracked in git — see .gitignore)
├── supabase/migrations/    # SQL migrations (pgvector, tables, hybrid_search fn)
├── src/
│   ├── core/               # config, logging
│   ├── db/                 # Supabase client
│   ├── ingestion/           # loader, cleaner, chunker
│   ├── embeddings/          # text embedding
│   ├── retrieval/           # LangChain retriever, cross-encoder reranker
│   ├── rag/                 # chain (production), pipeline (manual), cache, schemas
│   ├── evaluation/           # Recall@K, MRR, LLM-as-judge
│   └── api/                  # FastAPI app, auth, input validation
├── scripts/                 # ingestion + evaluation entry points
└── tests/                   # unit tests (no network calls)
```

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```dotenv
SUPABASE_URL=
SUPABASE_KEY=          # service_role / secret key — NOT the publishable key
GEMINI_API_KEY=
API_KEY=               # generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=MMCQA-RAG
```

### 3. Set up the database

Run the SQL files in `supabase/migrations/` **in order** (001 → 007) via the Supabase SQL Editor or `supabase db push`. This enables pgvector, creates the `documents` and `document_chunks` tables, and defines the `hybrid_search` function (semantic + full-text search combined via Reciprocal Rank Fusion).

### 4. Ingest the documents

```bash
python -m scripts.ingest_all
```

## Running

```bash
uvicorn src.api.main:app --reload
```

Interactive docs: `http://localhost:8000/docs` (use the "Authorize" button to set `X-API-Key`).

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{"question": "What is traditional Syrian clothing like?"}'
```

Response:

```json
{
  "answer": "...",
  "confidence": "high",
  "sources": [{"document_id": 4, "content": "...", "score": 0.04, "rerank_score": 7.2}],
  "usage": {"input_tokens": 493, "output_tokens": 146, "estimated_cost_usd": 0.0005}
}
```

## Evaluation

```bash
python -m scripts.evaluate           # retrieval quality: Recall@K, MRR
python -m scripts.evaluate_answers   # answer quality: faithfulness, relevance (LLM-as-judge)
```

Current results on the 10-document test corpus: **Recall@5 = 88.9%**, **MRR = 0.889**,
**Faithfulness = 100%**, **Relevance = 100%**.

## Testing

```bash
pytest tests/ -v
```

Covers chunking/cleaning logic, evaluation metrics, input validation
(including basic prompt-injection patterns), and caching. No network
calls — safe to run offline or in CI. All 36 tests passing.

## Security notes

- Uses Supabase's `service_role` key server-side only (bypasses RLS) — never expose this key to a client.
- `/ask` requires an `X-API-Key` header.
- Basic input validation rejects empty/oversized questions and a small set of known prompt-injection phrasings. This reduces but does not eliminate prompt-injection risk.

## Known limitations

- Embedding and re-ranking models are English-only — swap models if multilingual support is needed.
- The in-memory cache and rate limiter are per-process; a multi-instance deployment would need a shared backend (e.g. Redis) instead.
- Re-ranking improved overall MRR but can occasionally drop a correct document that was present before re-ranking (observed on one of nine test questions), likely due to the lightweight embedding/re-ranking models and topical overlap between source documents.