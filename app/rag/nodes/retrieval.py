"""
Node 2 — Retrieval

Performs a vector-similarity search against ChromaDB using the rewritten query
produced by Node 1. Returns the top-k most relevant document chunks with
their source metadata attached.
"""

import logging
from langchain_core.vectorstores import VectorStore

from app.rag.state import RAGState

logger = logging.getLogger(__name__)


def retrieval_node(state: RAGState, vectorstore: VectorStore, top_k: int = 5) -> dict:
    """
    Retrieve the top-k chunks most similar to the rewritten query.
    Uses the rewritten_query if available; falls back to the raw question.
    """
    query = state.get("rewritten_query") or state["question"]
    logger.info("Retrieving top-%d chunks for: %s", top_k, query)

    docs = vectorstore.similarity_search(query, k=top_k)

    logger.info("Retrieved %d documents", len(docs))
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        logger.debug("  [%d] %s — %s…", i, src, doc.page_content[:80])

    return {"documents": docs}
