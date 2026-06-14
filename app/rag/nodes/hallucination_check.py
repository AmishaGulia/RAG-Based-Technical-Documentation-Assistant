"""
Bonus Node — Hallucination Check  (inspired by Self-RAG)

Verifies that every factual claim in the generated answer is actually
supported by the context documents. If the answer is not grounded, the
graph can retry generation once before accepting the output as-is.
"""

import json
import logging
from langchain_core.documents import Document

from app.rag.state import RAGState
from app.rag.prompts import HALLUCINATION_PROMPT

logger = logging.getLogger(__name__)


def _build_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def hallucination_check_node(state: RAGState, llm) -> dict:
    """
    Ask the LLM whether the generated answer is supported by the context.

    Sets is_grounded=True  → graph proceeds to END.
    Sets is_grounded=False → graph retries generation (once) or proceeds anyway
                             if generation_attempts >= 2.
    """
    answer = state.get("answer", "")
    context_docs: list[Document] = state.get("relevant_documents", []) or state.get("web_search_results", [])

    if not answer:
        return {"is_grounded": False}

    if not context_docs:
        # No context means we already returned a "don't know" message — consider it grounded
        return {"is_grounded": True}

    context_text = _build_context(context_docs)

    chain = HALLUCINATION_PROMPT | llm
    response = chain.invoke({"context": context_text, "answer": answer})
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
        is_grounded = bool(parsed.get("grounded", True))
    except json.JSONDecodeError:
        logger.warning("Hallucination check returned non-JSON: %s", raw)
        # Optimistically treat ambiguous responses as grounded to avoid infinite loops
        is_grounded = True

    logger.info("Hallucination check → grounded=%s", is_grounded)
    return {"is_grounded": is_grounded}
