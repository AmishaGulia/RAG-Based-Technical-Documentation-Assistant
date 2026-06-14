"""
Document ingestion pipeline.

Orchestrates: load → chunk → embed → store

Chunking strategy:
  RecursiveCharacterTextSplitter with chunk_size=800, overlap=100.

  Rationale:
  - Technical documentation mixes prose (paragraphs), code (fenced blocks),
    and lists. The recursive splitter tries larger separators first
    ("\n\n", "\n", " ") so it preferentially splits on paragraph / section
    boundaries before breaking mid-sentence.
  - chunk_size=800 tokens (~600 words) keeps each chunk semantically coherent
    while fitting comfortably inside embedding model context windows.
  - overlap=100 ensures that sentences crossing chunk boundaries can still
    be retrieved; vital for code examples that span multiple lines.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore

from app.ingestion.loader import load_file, load_from_url, load_from_bytes

logger = logging.getLogger(__name__)

# Default directory to persist binary documents fetched from URLs (PDF, DOCX)
_DOWNLOADS_DIR = Path("data/downloads")

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],  # try coarse splits first
    length_function=len,
)


def _chunk_and_tag(docs: list[Document]) -> list[Document]:
    """
    Split each Document into chunks and annotate every chunk with
    chunk_index so the API can surface precise source references.
    """
    chunks: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        splits = SPLITTER.split_documents([doc])
        for idx, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["source"] = source  # propagate source
        chunks.extend(splits)
        logger.debug("  %s → %d chunks", source, len(splits))
    return chunks


def ingest_files(paths: list[Path], vectorstore: VectorStore) -> tuple[int, list[str]]:
    """
    Load local files, chunk, and upsert into the vector store.
    Supports all formats in loader._EXT_MAP (PDF, DOCX, MD, TXT, HTML, CSV, JSON…).
    Returns (chunks_added, source_names).
    """
    raw: list[Document] = []
    for path in paths:
        raw.extend(load_file(path))

    chunks = _chunk_and_tag(raw)
    if chunks:
        vectorstore.add_documents(chunks)
        logger.info("Ingested %d chunks from %d files", len(chunks), len(paths))

    sources = list({c.metadata["source"] for c in chunks})
    return len(chunks), sources


def ingest_urls(
    urls: list[str],
    vectorstore: VectorStore,
    save_dir: Path | None = _DOWNLOADS_DIR,
) -> tuple[int, list[str]]:
    """
    Fetch URLs, chunk, and upsert into the vector store.
    Content-type detection routes each URL to the correct parser (HTML, PDF, DOCX, …).
    Binary documents (PDF, DOCX) are saved to save_dir before parsing.
    Returns (chunks_added, source_urls).
    """
    raw: list[Document] = []
    for url in urls:
        raw.extend(load_from_url(url, save_dir=save_dir))

    chunks = _chunk_and_tag(raw)
    if chunks:
        vectorstore.add_documents(chunks)
        logger.info("Ingested %d chunks from %d URLs", len(chunks), len(urls))

    sources = list({c.metadata["source"] for c in chunks})
    return len(chunks), sources


def ingest_upload(content: bytes, filename: str, vectorstore: VectorStore) -> tuple[int, list[str]]:
    """
    Ingest an uploaded file (bytes). Format is detected from filename extension.
    Returns (chunks_added, [filename]).
    """
    raw = load_from_bytes(content, filename)
    chunks = _chunk_and_tag(raw)
    if chunks:
        vectorstore.add_documents(chunks)
        logger.info("Ingested %d chunks from upload: %s", len(chunks), filename)
    return len(chunks), [filename] if chunks else []
