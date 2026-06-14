"""
GET /documents — List all indexed documents in the vector store.

Returns unique source identifiers, chunk counts per source, and a short
content preview so callers can verify the corpus without calling /query.
"""

import logging
from collections import defaultdict
from fastapi import APIRouter, HTTPException

from app.api.schemas import DocumentsResponse, DocumentInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse, summary="List indexed documents")
async def list_documents():
    """
    Return metadata about every document currently in the vector store.
    Sources are grouped and sorted alphabetically.
    """
    from app.main import get_app_state

    vectorstore = get_app_state()["vectorstore"]

    try:
        # ChromaDB's underlying collection exposes .get() for direct access
        collection = vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.exception("Failed to list documents: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    raw_docs: list[str] = result.get("documents") or []
    raw_meta: list[dict] = result.get("metadatas") or []

    if not raw_docs:
        return DocumentsResponse(total_chunks=0, unique_sources=0, documents=[])

    # Group chunks by source
    source_chunks: dict[str, list[str]] = defaultdict(list)
    for content, meta in zip(raw_docs, raw_meta):
        source = meta.get("source", "unknown") if meta else "unknown"
        source_chunks[source].append(content)

    doc_infos: list[DocumentInfo] = [
        DocumentInfo(
            source=src,
            chunk_count=len(chunks),
            preview=chunks[0][:150] if chunks else "",
        )
        for src, chunks in sorted(source_chunks.items())
    ]

    return DocumentsResponse(
        total_chunks=len(raw_docs),
        unique_sources=len(doc_infos),
        documents=doc_infos,
    )
