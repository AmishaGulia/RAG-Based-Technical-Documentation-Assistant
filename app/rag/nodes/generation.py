"""
Node 4 — Generation

Uses the relevant (graded) document chunks as context to generate an answer.
When the web-search fallback has fired, web_search_results are merged into
the context instead.  Every cited source is extracted and returned as a
structured list so the API can surface them to the caller.
"""

import logging
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.rag.state import RAGState
from app.rag.prompts import GENERATION_PROMPT

logger = logging.getLogger(__name__)


def _format_chat_history(messages: list[BaseMessage]) -> str:
    if not messages:
        return "None"
    lines = []
    for msg in messages[-6:]:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _build_context(docs: list[Document]) -> str:
    """Render a list of Document chunks as a numbered context block for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", i)
        parts.append(
            f"[{i}] Source: {source} (chunk {chunk_idx})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[str]:
    """Deduplicate and return unique source identifiers."""
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def generation_node(state: RAGState, llm) -> dict:
    """
    Generate a grounded answer from the relevant context.
    Merges web search results when the normal retrieval path failed.
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])
    history_text = _format_chat_history(chat_history)

    # Prefer normally-graded docs; fall back to web results if the retrieval pipeline failed
    context_docs: list[Document] = state.get("relevant_documents", [])
    if not context_docs:
        context_docs = state.get("web_search_results", [])

    if not context_docs:
        # No context at all — return a safe "I don't know" rather than hallucinating
        answer = (
            "I could not find relevant information in the documentation or web search "
            "to answer your question. Please try rephrasing, or check the official docs directly."
        )
        return {
            "answer": answer,
            "sources": [],
            "is_grounded": True,        # trivially grounded — no false claims
            "generation_attempts": state.get("generation_attempts", 0) + 1,
        }

    context_text = _build_context(context_docs)
    sources = _extract_sources(context_docs)

    logger.info("Generating answer from %d context chunks", len(context_docs))

    chain = GENERATION_PROMPT | llm
    response = chain.invoke({
        "question": question,
        "context": context_text,
        "chat_history": history_text,
    })
    answer = response.content.strip()

    logger.info("Answer generated (%d chars)", len(answer))

    return {
        "answer": answer,
        "sources": sources,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }
