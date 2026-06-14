"""
Standalone document ingestion script.

Run this before starting the API server if you want to pre-populate the
vector store with the bundled corpus or your own files/URLs.

Usage:
    # Ingest the bundled docs/ corpus
    python scripts/ingest_docs.py

    # Ingest specific files
    python scripts/ingest_docs.py --files path/to/doc1.md path/to/doc2.md

    # Ingest from URLs
    python scripts/ingest_docs.py --urls https://fastapi.tiangolo.com/tutorial/

    # Clear the vector store and re-ingest
    python scripts/ingest_docs.py --reset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--files", nargs="+", type=Path, help="Local markdown / text files to ingest")
    parser.add_argument("--urls", nargs="+", help="URLs to fetch and ingest")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the vector store first")
    args = parser.parse_args()

    from app.core.config import get_settings
    settings = get_settings()

    # ── Build vector store ────────────────────────────────────────────────────
    from langchain_chroma import Chroma
    from app.core.embeddings import build_embeddings

    embeddings = build_embeddings(settings)

    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)

    if args.reset:
        logger.warning("--reset: deleting existing collection '%s'", settings.chroma_collection)
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    from app.ingestion.pipeline import ingest_files, ingest_urls

    total = 0

    # ── Ingest files ──────────────────────────────────────────────────────────
    files: list[Path] = args.files or []
    if not files and not args.urls:
        # Default: ingest everything in docs/
        docs_dir = Path(__file__).parent.parent / "docs"
        files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt")) +list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.docx"))
        logger.info("No arguments given — ingesting all files in docs/ (%d files)", len(files))

    if files:
        n, sources = ingest_files(files, vectorstore)
        logger.info("Files → %d chunks from %s", n, sources)
        total += n

    # ── Ingest URLs ───────────────────────────────────────────────────────────
    if args.urls:
        n, sources = ingest_urls(args.urls, vectorstore)
        logger.info("URLs → %d chunks from %s", n, sources)
        total += n

    logger.info("Ingestion complete. Total chunks added: %d", total)
    logger.info("Vector store now contains %d chunks", vectorstore._collection.count())


if __name__ == "__main__":
    main()
