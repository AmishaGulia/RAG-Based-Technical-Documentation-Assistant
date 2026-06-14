"""
POST /ingest — Add documents to the vector store.

Accepts:
  POST /ingest/urls  — JSON body with a list of URLs; auto-detects format via
                       Content-Type / extension (HTML, PDF, DOCX, plain text, …).
                       Binary documents are saved to data/downloads/ before parsing.
  POST /ingest/file  — Multipart upload for any supported format:
                       Markdown, plain text, HTML, PDF, DOCX, CSV, JSON / JSONL.

Both paths run through the same chunking + embedding pipeline.
"""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import IngestURLRequest, IngestResponse
from app.ingestion.pipeline import ingest_urls, ingest_upload

logger = logging.getLogger(__name__)
router = APIRouter()

# MIME types accepted for file upload — matches loader._EXT_MAP coverage
_ALLOWED_MIME_TYPES = {
    # Plain text variants
    "text/plain",
    "text/markdown",
    "text/x-rst",
    "text/x-markdown",
    # HTML
    "text/html",
    "application/xhtml+xml",
    # PDF
    "application/pdf",
    # Word documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    # CSV
    "text/csv",
    "application/csv",
    # JSON
    "application/json",
    "application/x-ndjson",
    "application/jsonl",
    # Browser fallback for .md, .csv, .jsonl etc. uploaded as binary
    "application/octet-stream",
}

# Human-readable label for the error message
_ALLOWED_LABEL = (
    "text/plain, text/markdown, text/html, application/pdf, "
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document, "
    "text/csv, application/json"
)

# Extensions we can handle regardless of the MIME type the browser reports
_ALLOWED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst",
    ".html", ".htm",
    ".pdf",
    ".docx", ".doc",
    ".csv",
    ".json", ".jsonl", ".ndjson",
}


@router.post(
    "/ingest/urls",
    response_model=IngestResponse,
    summary="Ingest from URLs",
    responses={
        422: {"description": "No extractable content found at the provided URLs"},
        500: {"description": "Internal server error during ingestion"},
    },
)
async def ingest_from_urls(request: IngestURLRequest):
    """
    Fetch and index documents from the provided URLs.

    The loader auto-detects each URL's format (HTML page, PDF, DOCX, plain text, …)
    via a HEAD request for Content-Type, falling back to the URL file extension.
    Binary documents (PDF, DOCX) are saved locally to data/downloads/ before parsing.
    """
    from app.main import get_app_state

    vectorstore = get_app_state()["vectorstore"]

    try:
        count, sources = ingest_urls(request.urls, vectorstore)
    except Exception as exc:
        logger.exception("URL ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if count == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable content found at the provided URLs.",
        )

    return IngestResponse(
        message=f"Successfully ingested {count} chunks from {len(sources)} source(s).",
        documents_added=count,
        sources=sources,
    )


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    summary="Ingest an uploaded file",
    responses={
        413: {"description": "File exceeds 20 MB limit"},
        415: {"description": "Unsupported file type or content-type"},
        422: {"description": "No content could be extracted from the file"},
        500: {"description": "Internal server error during ingestion"},
        501: {"description": "Optional dependency not installed (pypdf / python-docx)"},
    },
)
async def ingest_file(file: Annotated[UploadFile, File(...)]):
    """
    Upload a document for indexing.

    Supported formats: Markdown, plain text, HTML, PDF, DOCX, CSV, JSON, JSONL.
    Maximum file size: 20 MB. Format is inferred from the filename extension first;
    the Content-Type header is used as a secondary signal.
    """
    from app.main import get_app_state

    vectorstore = get_app_state()["vectorstore"]

    filename = file.filename or "upload.bin"
    extension = Path(filename).suffix.lower()

    # Accept if the extension is known; fall back to MIME type check
    if extension not in _ALLOWED_EXTENSIONS:
        content_type = (file.content_type or "").split(";")[0].strip()
        if content_type not in _ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{extension}' / content-type '{content_type}'. "
                    f"Accepted formats: {_ALLOWED_LABEL}."
                ),
            )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20 MB guard
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")

    try:
        count, sources = ingest_upload(content, filename, vectorstore)
    except ImportError as exc:
        # Missing optional dependency (pypdf, python-docx)
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        logger.exception("File ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(exc))

    if count == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No content could be extracted from '{filename}'.",
        )

    return IngestResponse(
        message=f"Successfully ingested {count} chunks from '{filename}'.",
        documents_added=count,
        sources=sources,
    )
