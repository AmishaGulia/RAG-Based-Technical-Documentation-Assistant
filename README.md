# RAG-Based Technical Documentation Assistant

A **self-corrective Retrieval-Augmented Generation** system built with LangGraph, ChromaDB, and FastAPI. It answers natural-language questions against a curated technical documentation corpus, with document grading, hallucination checking, web-search fallback, and conversation memory.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────┐
│  Node 1: Query      │  Rewrites query for better retrieval;
│  Analysis           │  classifies type (conceptual / how-to /
└────────┬────────────┘  troubleshooting / api-reference)
         │
         ▼
┌─────────────────────┐
│  Node 2: Retrieval  │  Vector similarity search in ChromaDB
│                     │  Returns top-k chunks + source metadata
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 3: Document   │  LLM grades each chunk: relevant / irrelevant
│  Grading ★          │  Filters irrelevant chunks
└────────┬────────────┘
         │
    ─────────────────── conditional edge ───────────────────────
    │                                                           │
    │ some relevant                               all irrelevant
    ▼                                                           ▼
┌─────────────────────┐                         retry_count < max?
│  Node 4: Generation │                        ┌──yes──┐  ┌──no──┐
│                     │                        │       │  │      │
└────────┬────────────┘               increment retry  │  web_search
         │                            ↓ query_analysis │     ↓
         ▼                            ↓ retrieval      │  generation
┌─────────────────────┐               ↓ grading ◄──────┘     │
│  Bonus: Hallucina-  │                                 hallucination_check
│  tion Check ★       │                                       │
└────────┬────────────┘                                      END
         │
    grounded? ──no (1st time)──► generation (retry once)
         │
        yes
         │
        END
```

**Key design decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM provider | Anthropic / OpenAI / Groq (configurable) | Swap via `.env` without code changes |
| Utility model | `claude-haiku-4-5-20251001` | Fast + cheap for grading, query analysis, hallucination check |
| Generation model | Configurable (same or higher quality) | Can upgrade to Sonnet for better answers |
| Embeddings | HuggingFace / OpenAI / Cohere / Google (configurable) | Default is local sentence-transformers (no API key); swap provider via `EMBEDDING_PROVIDER` |
| Vector store | ChromaDB (local persistence) | Zero infrastructure, sub-ms queries, LangChain native |
| Chunking | `RecursiveCharacterTextSplitter`, size=800, overlap=100 | Technical docs have paragraphs + code blocks — recursive splitter respects these boundaries |
| Ingestion formats | PDF, DOCX, MD, TXT, HTML, CSV, JSON, JSONL | Format detected from file extension or HTTP Content-Type; each has a distinct extractor |
| Retry limit | 2 retries before web fallback | Balances thoroughness vs latency |
| State design | TypedDict with custom reducer for `chat_history` | Append-only semantics prevents message loss in multi-node writes |

---

## Project Structure

```
.
├── app/
│   ├── main.py                      # FastAPI app + startup lifespan
│   ├── core/
│   │   ├── config.py                # All settings via pydantic-settings
│   │   └── embeddings.py            # Embedding model factory (provider-aware)
│   ├── api/
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── routes/
│   │       ├── query.py             # POST /query
│   │       ├── ingest.py            # POST /ingest/urls, POST /ingest/file
│   │       ├── documents.py         # GET /documents
│   │       └── feedback.py          # POST /feedback
│   ├── rag/
│   │   ├── state.py                 # LangGraph RAGState TypedDict
│   │   ├── prompts.py               # All LLM prompt templates
│   │   ├── graph.py                 # StateGraph wiring + routing functions
│   │   └── nodes/
│   │       ├── query_analysis.py    # Node 1: query rewrite + classification
│   │       ├── retrieval.py         # Node 2: vector similarity search
│   │       ├── grading.py           # Node 3: LLM relevance grading
│   │       ├── generation.py        # Node 4: answer generation with citations
│   │       ├── hallucination_check.py  # Bonus: factual grounding check
│   │       └── web_search.py        # Bonus: Tavily web search fallback
│   ├── ingestion/
│   │   ├── loader.py                # Format-aware loaders (PDF, DOCX, MD, HTML, CSV, JSON…)
│   │   └── pipeline.py             # Load → chunk → embed → store
│   └── memory/
│       └── session_store.py         # Bonus: in-memory conversation history
├── docs/                            # Bundled documentation corpus (9 files, 3 formats)
│   ├── fastapi_guide.md             #   Markdown guides
│   ├── langgraph_guide.md
│   ├── langchain_guide.md
│   ├── pydantic_guide.md
│   ├── chromadb_guide.md
│   ├── python_async_guide.md
│   ├── rag_paper_lewis2020.pdf      #   PDFs (arxiv)
│   ├── self_rag_paper_asai2023.pdf
│   └── taxii_rest_api_spec.docx     #   DOCX (OASIS)
├── frontend/
│   └── streamlit_app.py             # Bonus: Streamlit chat UI
├── scripts/
│   └── ingest_docs.py               # Standalone ingestion CLI
├── data/
│   ├── chroma_db/                   # ChromaDB persistence (auto-created)
│   ├── downloads/                   # Binary docs fetched from URLs (auto-created)
│   └── feedback/                    # JSONL feedback files (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Document Corpus

The `docs/` folder ships with **9 files across three formats** covering the core stack and foundational research behind this project.

### Markdown guides (hand-written)

| File | Topics |
|------|--------|
| `fastapi_guide.md` | Routing, path/query params, request body, dependencies, middleware, background tasks, file uploads, lifespan |
| `langgraph_guide.md` | StateGraph, nodes, edges, conditional edges, checkpointing, human-in-the-loop, streaming, retry patterns |
| `langchain_guide.md` | LCEL, chat models, prompt templates, output parsers, document loaders, text splitters, embeddings, vector stores, RAG chains |
| `pydantic_guide.md` | BaseModel, Field constraints, validators, nested models, serialisation, settings management, generic models |
| `chromadb_guide.md` | Collections, adding documents, querying, filtering, embedding functions, LangChain integration, performance tips |
| `python_async_guide.md` | async/await, event loops, asyncio tasks, concurrency patterns, FastAPI async routes |

### PDFs (downloaded from arxiv)

| File | Source | Topics |
|------|--------|--------|
| `rag_paper_lewis2020.pdf` | [arxiv 2005.11401](https://arxiv.org/abs/2005.11401) | Original RAG paper — dense retrieval + seq2seq generation; the technique this system is built on |
| `self_rag_paper_asai2023.pdf` | [arxiv 2310.11511](https://arxiv.org/abs/2310.11511) | Self-RAG — adaptive retrieval, self-reflection tokens, factual grounding; inspiration for the hallucination check node |

### DOCX (downloaded from OASIS)

| File | Source | Topics |
|------|--------|--------|
| `taxii_rest_api_spec.docx` | [OASIS TAXII v2.1](https://docs.oasis-open.org/cti/taxii/v2.1/os/) | TAXII REST API specification — API design patterns, HTTP endpoints, authentication, resource schemas |

These files are also a **live demo of the multi-format ingestion pipeline**: the PDF and DOCX are parsed by their dedicated extractors (`pypdf`, `python-docx`) on first ingest, alongside the Markdown guides processed as plain text.

---

## Supported Document Formats

The ingestion pipeline auto-detects format from the file extension (uploads / local files) or HTTP `Content-Type` header (URLs), then routes to a dedicated extractor:

| Format | Extensions / MIME | Extractor behaviour |
|--------|-------------------|---------------------|
| **PDF** | `.pdf` / `application/pdf` | `pypdf` — one `Document` per page; `page` number in metadata |
| **Word** | `.docx`, `.doc` / `application/vnd.openxmlformats…` | `python-docx` — paragraphs + table cells |
| **Markdown** | `.md`, `.markdown` / `text/markdown` | Plain-text passthrough |
| **Plain text** | `.txt`, `.rst` / `text/plain` | Plain-text passthrough |
| **HTML** | `.html`, `.htm` / `text/html` | BeautifulSoup — prefers `<main>` / `<article>` containers |
| **CSV** | `.csv` / `text/csv` | Each row → `key: value` line (capped at 500 rows) |
| **JSON** | `.json` / `application/json` | Pretty-printed JSON object |
| **JSONL** | `.jsonl`, `.ndjson` / `application/x-ndjson` | One record per line (capped at 200 records) |

Unknown extensions fall back to plain-text.

### URL ingestion with format detection

`POST /ingest/urls` (and `python scripts/ingest_docs.py --urls`) now handles any URL, not just HTML pages:

1. **HEAD request** reads `Content-Type` — no full download until the format is confirmed.
2. Routes to the matching extractor (same table above).
3. **Binary documents** (PDF, DOCX) are saved to `data/downloads/` before parsing so they can be re-ingested without re-fetching.
4. Falls back to URL file extension if the server doesn't set `Content-Type`.

```bash
# HTML page (existing behaviour)
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://fastapi.tiangolo.com/tutorial/"]}'

# PDF hosted at a URL — downloaded, saved, and parsed automatically
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://arxiv.org/pdf/2310.01234.pdf"]}'
```

---

## Setup

### 1. Clone and create virtual environment

```bash
cd cld-a14

python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` sentence-transformer model (~85 MB). It is cached locally after that.

### 3. Configure environment

```bash
cp .env.example .env
```

**Minimum required** — set your LLM provider key:

```dotenv
LLM_PROVIDER=anthropic          # or openai | groq
ANTHROPIC_API_KEY=sk-ant-...
```

**Embedding provider** (optional — defaults to local HuggingFace, no key needed):

```dotenv
# Switch to OpenAI embeddings:
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# Switch to Cohere:
EMBEDDING_PROVIDER=cohere
EMBEDDING_MODEL=embed-english-v3.0
COHERE_API_KEY=...

# Switch to Google:
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=models/text-embedding-004
GOOGLE_API_KEY=...
```

> **Important:** changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` after documents are ingested requires a full re-ingest — vector spaces are incompatible across models:
> ```bash
> python scripts/ingest_docs.py --reset
> ```

**LLM providers:** `anthropic` | `openai` | `groq`  
**Embedding providers:** `huggingface` (default, local) | `openai` | `cohere` | `google`

```dotenv
# Optional — for web search fallback
TAVILY_API_KEY=tvly-...
```

### 4. (Optional) Pre-ingest the corpus

The server auto-ingests `docs/` on first startup if the vector store is empty. To ingest manually or reset:

```bash
# Ingest bundled docs/
python scripts/ingest_docs.py

# Ingest specific files
python scripts/ingest_docs.py --files my_doc.md

# Ingest from URLs
python scripts/ingest_docs.py --urls https://fastapi.tiangolo.com/tutorial/

# Reset and re-ingest
python scripts/ingest_docs.py --reset
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### 6. (Optional) Start the Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

---

## API Reference

### `POST /query` — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add middleware in FastAPI?"}'
```

**Response:**

```json
{
  "answer": "You can add middleware in FastAPI using the `@app.middleware('http')` decorator...\n[Source: fastapi_guide.md]",
  "sources": [
    {
      "source": "fastapi_guide.md",
      "chunk_index": 7,
      "content_preview": "## Middleware\n\nAdd middleware to execute code before/after every request..."
    }
  ],
  "query_type": "how-to",
  "rewritten_query": "how to add and configure HTTP middleware in FastAPI ASGI application",
  "is_grounded": true,
  "web_search_used": false,
  "retry_count": 0,
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

### Follow-up question (conversation memory)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I use it for authentication?", "session_id": "3fa85f64-..."}'
```

### `POST /ingest/urls` — Ingest from URLs (any format)

Format is auto-detected via `Content-Type` / URL extension. Binary docs are saved to `data/downloads/`.

```bash
# HTML page
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://fastapi.tiangolo.com/tutorial/first-steps/"]}'

# PDF at a URL
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/report.pdf"]}'
```

### `POST /ingest/file` — Upload a file

Accepts: `.md`, `.txt`, `.rst`, `.html`, `.pdf`, `.docx`, `.csv`, `.json`, `.jsonl` (max 20 MB).

```bash
curl -X POST http://localhost:8000/ingest/file -F "file=@report.pdf"
curl -X POST http://localhost:8000/ingest/file -F "file=@data.csv"
curl -X POST http://localhost:8000/ingest/file -F "file=@my_doc.md"
```

### `GET /documents` — List indexed corpus

```bash
curl http://localhost:8000/documents
```

**Response:**

```json
{
  "total_chunks": 147,
  "unique_sources": 5,
  "documents": [
    {
      "source": "fastapi_guide.md",
      "chunk_count": 34,
      "preview": "# FastAPI — Complete Guide\n\n## What is FastAPI?..."
    }
  ]
}
```

### `POST /feedback` — Submit feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "3fa85f64-...",
    "question": "How do I add middleware?",
    "answer": "You can add middleware using...",
    "rating": "thumbs_up",
    "comment": "Clear and accurate!"
  }'
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "indexed_chunks": 147}
```

---

## Bonus Features

### Hallucination Check (Self-RAG inspired)

After generation, a separate LLM call verifies that every factual claim in the answer is supported by the retrieved context. If the check fails, the graph retries generation once before accepting the output. The `is_grounded` field in the response reports the result.

### Web Search Fallback (Tavily)

When the vector store returns no relevant documents after `MAX_RETRIES` query rewrites, the graph automatically falls back to Tavily web search. Results are treated as additional context documents and passed to the generation node. Set `ENABLE_WEB_SEARCH=false` to disable. `TAVILY_API_KEY` is required.

### Conversation Memory

Pass the `session_id` from one `/query` response into the next request to maintain context across follow-up questions. Each session stores a rolling window of the last 3 question-answer turns (6 messages) to stay within model context limits.

### Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

Features: chat history, source citations panel, feedback buttons, URL ingestion, corpus browser, new-conversation button.

---

## Design Decisions & Tradeoffs

### What I would improve with more time

1. **Persistent session storage** — Current in-memory store loses history on restart. Would replace with Redis or SQLite.
2. **Hybrid search** — Combine BM25 (keyword) with vector similarity for better recall on exact API names.
3. **Chunk-level metadata** — Store section headings in metadata and include them in the retrieval context for better citations.
4. **Streaming responses** — Use `graph.astream_events()` and FastAPI SSE to stream tokens to the client.
5. **Re-ranking** — Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and grading for higher precision.
6. **Async nodes** — Convert all LLM calls to async for better concurrency under load.
7. **Evaluation** — Add RAGAS-based evaluation (faithfulness, answer relevancy, context recall) to benchmark changes.

### Chunking strategy

`RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`.

The recursive splitter tries `["\n\n", "\n", ". ", " ", ""]` in order, so it splits on paragraph boundaries before breaking sentences. This is important for technical docs where a code block immediately following prose is semantically one unit. Overlap of 100 characters ensures sentences and short code snippets that straddle boundaries are retrievable from either adjacent chunk.

### Embedding provider tradeoffs

| Provider | Model | Dims | API key | Tradeoff |
|----------|-------|------|---------|----------|
| **HuggingFace** (default) | `all-MiniLM-L6-v2` | 384 | None | Free, local, ~85 MB download; slightly lower recall on domain jargon |
| **OpenAI** | `text-embedding-3-small` | 1536 | Required | Best quality for English technical text; pay-per-use |
| **Cohere** | `embed-english-v3.0` | 1024 | Required | Strong multilingual support; good for mixed-language corpora |
| **Google** | `text-embedding-004` | 768 | Required | Competitive quality; integrates naturally with Gemini LLM stack |

All providers use the same ChromaDB backend. To switch, set `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` in `.env`, then run `python scripts/ingest_docs.py --reset` to rebuild the vector store.

### State schema design

`retry_count` and `generation_attempts` are tracked in state (not external variables) so the graph is deterministic and replayable from any checkpoint. The `all_irrelevant` boolean is a clear signal for the conditional edge router rather than checking `len(relevant_documents) == 0` in the router function itself.

---

## Assumptions

1. Documents are English-language technical text.
2. The corpus fits in memory for embedding (each file < 10 MB).
3. One ChromaDB collection is sufficient (no multi-tenancy).
4. Session IDs are not authenticated — suitable for single-user or trusted environments.
