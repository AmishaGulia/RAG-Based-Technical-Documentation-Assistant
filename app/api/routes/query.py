"""
POST /query — Submit a natural language question.

Initialises LangGraph state, runs the compiled graph, updates conversation
memory, and returns the structured answer with sources.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceDoc
from app.core.config import get_settings
from app.memory import session_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query(request: QueryRequest, app_state: dict = Depends(lambda: None)):
    """
    Run the RAG pipeline for a user question.

    - Rewrites and classifies the query
    - Retrieves relevant documentation chunks
    - Grades document relevance (self-corrective)
    - Generates a grounded answer with citations
    - Checks for hallucinations (bonus)

    Supports follow-up questions via session_id.
    """
    from app.main import get_app_state   # imported here to avoid circular import

    state_container = get_app_state()
    graph = state_container["graph"]
    settings = get_settings()

    # ── Session / conversation memory ──────────────────────────────────────────
    session_id = session_store.get_or_create_session(request.session_id)
    chat_history = session_store.get_history(session_id)

    # ── Build initial graph state ──────────────────────────────────────────────
    initial_state = {
        "question": request.question,
        "session_id": session_id,
        "chat_history": chat_history,
        "rewritten_query": "",
        "query_type": "conceptual",
        "documents": [],
        "relevant_documents": [],
        "web_search_results": [],
        "web_search_used": False,
        "answer": "",
        "sources": [],
        "is_grounded": False,
        "all_irrelevant": False,
        "retry_count": 0,
        "generation_attempts": 0,
    }

    try:
        logger.info("Running RAG graph for session %s: %s", session_id, request.question)
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # ── Persist conversation turn ──────────────────────────────────────────────
    session_store.append_turn(session_id, request.question, result.get("answer", ""))

    # ── Build structured sources list ──────────────────────────────────────────
    relevant_docs = result.get("relevant_documents", []) or result.get("web_search_results", [])
    sources: list[SourceDoc] = []
    seen: set[tuple] = set()
    for doc in relevant_docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            sources.append(SourceDoc(
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                content_preview=doc.page_content[:200],
            ))

    return QueryResponse(
        answer=result.get("answer", "No answer generated."),
        sources=sources,
        query_type=result.get("query_type", "conceptual"),
        rewritten_query=result.get("rewritten_query", request.question),
        is_grounded=result.get("is_grounded", True),
        web_search_used=result.get("web_search_used", False),
        retry_count=result.get("retry_count", 0),
        session_id=session_id,
    )
