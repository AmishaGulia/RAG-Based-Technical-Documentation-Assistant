"""
LangGraph state schema.

The RAGState TypedDict is the single object that flows through every node in
the graph. Each field is either set at graph entry (question, session_id) or
produced by a specific node. Fields are cumulative — later nodes read from
fields written by earlier ones without wiping them.

Flow overview:
  query_analysis → retrieval → document_grading
       ↑ (retry)                       ↓ (conditional)
       └──── query_rewrite ←───── all irrelevant?
                                        ↓ (no more retries)
                                   web_search (bonus)
                                        ↓
                                   generation
                                        ↓
                                 hallucination_check (bonus)
                                        ↓
                                       END
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

# Reducer for chat_history: append new messages rather than overwrite the list.
def _append_messages(existing: list, new: list) -> list:
    return existing + new


class RAGState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str                              # Original user question
    session_id: str                            # Identifies the conversation session

    # ── Conversation memory (bonus) ───────────────────────────────────────────
    # Uses a custom reducer so multiple nodes can append without clobbering.
    chat_history: Annotated[list[BaseMessage], _append_messages]

    # ── Query analysis node output ────────────────────────────────────────────
    rewritten_query: str                       # Expanded / clarified query for retrieval
    query_type: str                            # conceptual | how-to | troubleshooting | api-reference

    # ── Retrieval node output ─────────────────────────────────────────────────
    documents: list[Document]                  # Raw top-k results from vector store

    # ── Document grading node output ──────────────────────────────────────────
    relevant_documents: list[Document]         # Subset of documents graded as relevant
    all_irrelevant: bool                       # True when zero chunks passed grading

    # ── Web search fallback (bonus) ───────────────────────────────────────────
    web_search_results: list[Document]         # Results from Tavily when vector store fails
    web_search_used: bool                      # Tracks whether fallback was triggered

    # ── Generation node output ────────────────────────────────────────────────
    answer: str                                # Final answer text
    sources: list[str]                         # Source file/URL identifiers cited

    # ── Hallucination check (bonus) ───────────────────────────────────────────
    is_grounded: bool                          # True if answer is supported by context
    generation_attempts: int                   # Guards against infinite hallucination retry

    # ── Control flow ──────────────────────────────────────────────────────────
    retry_count: int                           # Number of query-rewrite retries so far
