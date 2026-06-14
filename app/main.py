"""
FastAPI application entry point.

Startup sequence:
  1. Load settings from .env
  2. Initialise the embedding model (local sentence-transformers, no API key)
  3. Connect to / create the ChromaDB vector store
  4. Build the LLM client(s) from the configured provider
  5. Compile the LangGraph workflow
  6. Auto-ingest the bundled docs/ corpus if the vector store is empty

All heavyweight objects are stored in the app.state dict and accessed via
get_app_state() so routes don't need to re-initialise them on every request.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import query, ingest, documents, feedback

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Module-level container — populated during startup
_app_state: dict = {}


def get_app_state() -> dict:
    """Return the global app state dict (graph, vectorstore, etc.)."""
    return _app_state


def _build_llm(settings):
    """
    Instantiate the LLM client based on LLM_PROVIDER env var.
    Returns (utility_llm, generation_llm) — may be the same object.
    """
    provider = settings.llm_provider.lower()
    logger.info("Building LLM — provider: %s, model: %s", provider, settings.llm_model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        utility_llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,          # deterministic for grading / analysis
            max_tokens=1024,
        )
        generation_llm = ChatAnthropic(
            model=settings.generation_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,        # slight creativity for answers
            max_tokens=2048,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        utility_llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        generation_llm = ChatOpenAI(
            model=settings.generation_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        utility_llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
        generation_llm = ChatGroq(
            model=settings.generation_model,
            api_key=settings.groq_api_key,
            temperature=0.2,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose anthropic | openai | groq")

    return utility_llm, generation_llm


def _build_vectorstore(settings):
    """
    Create or connect to the ChromaDB persistent vector store.
    Embedding provider is selected by EMBEDDING_PROVIDER (see app/core/embeddings.py).
    """
    from langchain_chroma import Chroma
    from app.core.embeddings import build_embeddings

    embeddings = build_embeddings(settings)

    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)

    logger.info("Connecting to ChromaDB at: %s", persist_dir)
    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def _auto_ingest_corpus(vectorstore, settings) -> None:
    """
    If the vector store is empty, automatically ingest the bundled docs/ corpus.
    This ensures the application works out-of-the-box without manual setup.
    """
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = 0

    if count > 0:
        logger.info("Vector store already has %d chunks — skipping auto-ingest", count)
        return

    docs_dir = Path(__file__).parent.parent / "docs"
    if not docs_dir.exists():
        logger.warning("docs/ directory not found — skipping auto-ingest")
        return

    md_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    if not md_files:
        logger.warning("No .md / .txt files in docs/ — skipping auto-ingest")
        return

    logger.info("Auto-ingesting %d corpus files from docs/…", len(md_files))
    from app.ingestion.pipeline import ingest_files
    n, sources = ingest_files(md_files, vectorstore)
    logger.info("Auto-ingest complete: %d chunks from %s", n, sources)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: build all shared objects once at startup."""
    settings = get_settings()

    vectorstore = _build_vectorstore(settings)
    utility_llm, generation_llm = _build_llm(settings)

    from app.rag.graph import build_graph
    graph = build_graph(utility_llm, generation_llm, vectorstore, settings)

    _app_state.update({
        "vectorstore": vectorstore,
        "utility_llm": utility_llm,
        "generation_llm": generation_llm,
        "graph": graph,
        "settings": settings,
    })

    _auto_ingest_corpus(vectorstore, settings)

    logger.info("RAG Documentation Assistant ready on %s:%d", settings.app_host, settings.app_port)
    yield
    # Cleanup (ChromaDB auto-persists; nothing else to close)
    logger.info("Shutting down")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description=(
        "A self-corrective Retrieval-Augmented Generation system built with "
        "LangGraph, ChromaDB, and FastAPI. Answers questions from a curated "
        "technical documentation corpus with citations and hallucination checking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────────────
app.include_router(query.router, tags=["RAG Query"])
app.include_router(ingest.router, tags=["Document Ingestion"])
app.include_router(documents.router, tags=["Document Management"])
app.include_router(feedback.router, tags=["Feedback"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "RAG Technical Documentation Assistant",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    try:
        count = _app_state["vectorstore"]._collection.count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}
