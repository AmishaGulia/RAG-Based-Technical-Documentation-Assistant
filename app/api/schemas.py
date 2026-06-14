"""
Pydantic request/response schemas for all FastAPI endpoints.
All schemas use strict typing with examples for auto-generated OpenAPI docs.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


# ── /query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to answer from the documentation.",
        examples=["How do I define path parameters in FastAPI?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory. Created automatically if omitted.",
    )

    model_config = {"json_schema_extra": {"examples": [{"question": "How do I define path parameters in FastAPI?"}]}}


class SourceDoc(BaseModel):
    """A single cited source chunk returned alongside the answer."""
    source: str = Field(description="File name or URL of the source document.")
    chunk_index: int = Field(description="Index of the chunk within the source document.")
    content_preview: str = Field(description="First 200 characters of the chunk for reference.")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer grounded in retrieved documentation.")
    sources: list[SourceDoc] = Field(default_factory=list, description="Documents cited in the answer.")
    query_type: str = Field(description="Classified query type (conceptual/how-to/troubleshooting/api-reference).")
    rewritten_query: str = Field(description="Expanded query used for retrieval.")
    is_grounded: bool = Field(description="Whether the answer passed the hallucination check.")
    web_search_used: bool = Field(default=False, description="True when fallback web search was triggered.")
    retry_count: int = Field(default=0, description="Number of retrieval retries performed.")
    session_id: str = Field(description="Session ID for follow-up questions.")


# ── /ingest ────────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="List of URLs to fetch and ingest as documentation.",
        examples=[["https://fastapi.tiangolo.com/tutorial/first-steps/"]],
    )


class IngestResponse(BaseModel):
    message: str
    documents_added: int = Field(description="Number of new chunks added to the vector store.")
    sources: list[str] = Field(description="Source identifiers that were ingested.")


# ── /documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    source: str
    chunk_count: int
    preview: str = Field(description="First 150 chars of the first chunk.")


class DocumentsResponse(BaseModel):
    total_chunks: int
    unique_sources: int
    documents: list[DocumentInfo]


# ── /feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str = Field(description="Session ID from a prior /query response.")
    question: str = Field(description="The question that was asked.")
    answer: str = Field(description="The answer that was received.")
    rating: Literal["thumbs_up", "thumbs_down"] = Field(description="Binary quality signal.")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional free-text comment.")


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str
