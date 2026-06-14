"""
Node 1 — Query Analysis

Rewrites the user's raw question into a retrieval-optimised query and
classifies it into one of four types to guide downstream processing.
Also handles query rewriting after a failed retrieval (retry path).
"""

import json
import logging

from langchain_core.messages import BaseMessage

from app.rag.state import RAGState
from app.rag.prompts import QUERY_ANALYSIS_PROMPT, QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


def _format_chat_history(messages: list[BaseMessage]) -> str:
    """Render chat history as a plain-text string for the prompt."""
    if not messages:
        return "None"
    lines = []
    for msg in messages[-6:]:  # only the last 3 turns (6 messages) to stay within token budget
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def query_analysis_node(state: RAGState, llm) -> dict:
    """
    Rewrites the question and classifies its type.
    Runs both on the first pass and on the retry path (with a different rewrite strategy).
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    chat_history = state.get("chat_history", [])
    history_text = _format_chat_history(chat_history)

    if retry_count == 0:
        # First pass — standard query expansion
        logger.info("Query analysis (initial): %s", question)
        chain = QUERY_ANALYSIS_PROMPT | llm
        response = chain.invoke({
            "question": question,
            "chat_history": history_text,
        })
        raw = response.content.strip()
        try:
            parsed = json.loads(raw)
            rewritten_query = parsed.get("rewritten_query", question)
            query_type = parsed.get("query_type", "conceptual")
        except json.JSONDecodeError:
            # If the LLM did not return valid JSON, fall back gracefully
            logger.warning("Query analysis returned non-JSON: %s", raw)
            rewritten_query = question
            query_type = "conceptual"
    else:
        # Retry path — use a different rewriting strategy to escape the retrieval dead-end
        logger.info("Query rewrite (retry %d): %s", retry_count, question)
        previous_query = state.get("rewritten_query", question)
        chain = QUERY_REWRITE_PROMPT | llm
        response = chain.invoke({
            "question": question,
            "previous_query": previous_query,
            "retry_count": retry_count,
        })
        rewritten_query = response.content.strip()
        query_type = state.get("query_type", "conceptual")  # preserve prior classification

    logger.info("Rewritten query: %s  |  type: %s", rewritten_query, query_type)
    return {
        "rewritten_query": rewritten_query,
        "query_type": query_type,
    }
