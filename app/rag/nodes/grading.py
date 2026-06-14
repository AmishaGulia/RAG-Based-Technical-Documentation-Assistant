"""
Node 3 — Document Grading  (self-corrective component)

Uses an LLM to judge whether each retrieved chunk is actually relevant to the
question. Irrelevant chunks are filtered out. When zero chunks survive, the
graph routes to the retry / web-search fallback path.
"""

import json
import logging
from langchain_core.documents import Document

from app.rag.state import RAGState
from app.rag.prompts import GRADING_PROMPT

logger = logging.getLogger(__name__)


def document_grading_node(state: RAGState, llm) -> dict:
    """
    Grade every retrieved document chunk as relevant or irrelevant.

    Returns:
        relevant_documents: list of chunks that passed the relevance check.
        all_irrelevant: True when the relevant list is empty (triggers retry).
    """
    question = state["question"]
    documents: list[Document] = state.get("documents", [])

    if not documents:
        logger.warning("No documents to grade — marking all_irrelevant=True")
        return {"relevant_documents": [], "all_irrelevant": True}

    chain = GRADING_PROMPT | llm
    relevant: list[Document] = []

    for doc in documents:
        response = chain.invoke({
            "question": question,
            "document": doc.page_content,
        })
        raw = response.content.strip()
        try:
            result = json.loads(raw)
            is_relevant = result.get("relevant", False)
        except json.JSONDecodeError:
            # Treat ambiguous LLM output as irrelevant to be conservative
            logger.warning("Grading returned non-JSON: %s", raw)
            is_relevant = False

        src = doc.metadata.get("source", "unknown")
        logger.debug("  Graded [%s] → relevant=%s", src, is_relevant)

        if is_relevant:
            relevant.append(doc)

    all_irrelevant = len(relevant) == 0
    logger.info(
        "Grading complete: %d/%d chunks relevant, all_irrelevant=%s",
        len(relevant), len(documents), all_irrelevant,
    )
    return {"relevant_documents": relevant, "all_irrelevant": all_irrelevant}
