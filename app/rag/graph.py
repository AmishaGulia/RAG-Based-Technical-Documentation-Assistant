"""
LangGraph StateGraph — wires all nodes together and defines routing logic.

Graph structure:
                        ┌─────────────────────────────────┐
                        │          query_analysis          │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │             retrieval            │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │         document_grading         │
                        └────────────────┬────────────────┘
                              (conditional edge)
                  ┌─────────────────────┴────────────────────┐
         relevant │                                          │ all irrelevant
                  ▼                                          ▼
          ┌──────────────┐                 retry_count < max_retries?
          │  generation  │                ┌──yes──┐       ┌──no──┐
          └──────┬───────┘                │       │       │      │
                 │                 query_analysis  │  web_search  │
         ┌───────▼────────┐        (retry path)   │       ▼      │
         │hallucination   │                        │  generation  │
         │    check       │                        └──────┬───────┘
         └───────┬────────┘                       hallucination_check
           (conditional)                                  │
      grounded?  │  not grounded                         END
        ┌────────┴────────┐
       END          generation (once more)
                          │
                   hallucination_check → END
"""

import functools
import logging
from typing import Literal

from langchain_core.vectorstores import VectorStore
from langgraph.graph import StateGraph, START, END

from app.core.config import Settings
from app.rag.state import RAGState
from app.rag.nodes.query_analysis import query_analysis_node
from app.rag.nodes.retrieval import retrieval_node
from app.rag.nodes.grading import document_grading_node
from app.rag.nodes.generation import generation_node
from app.rag.nodes.hallucination_check import hallucination_check_node
from app.rag.nodes.web_search import web_search_node

logger = logging.getLogger(__name__)


# ── Routing functions (conditional edges) ─────────────────────────────────────

def route_after_grading(state: RAGState, max_retries: int) -> Literal[
    "generation", "query_analysis", "web_search"
]:
    """
    Decides what happens after document grading:
    - Relevant docs exist  → generate an answer
    - All irrelevant + retries remaining → rewrite query and re-retrieve
    - All irrelevant + no retries left   → web search fallback
    """
    if not state.get("all_irrelevant", False):
        return "generation"

    retry_count = state.get("retry_count", 0)
    if retry_count < max_retries:
        logger.info("All irrelevant — retrying (attempt %d/%d)", retry_count + 1, max_retries)
        return "query_analysis"

    logger.info("Max retries reached — falling back to web search")
    return "web_search"


def route_after_hallucination_check(state: RAGState) -> Literal["generation", "__end__"]:
    """
    If the answer is not grounded AND we haven't retried generation yet,
    retry once. Otherwise accept the answer (even if not perfect) to break
    any potential loop.
    """
    is_grounded = state.get("is_grounded", True)
    generation_attempts = state.get("generation_attempts", 1)

    if is_grounded or generation_attempts >= 2:
        return "__end__"

    logger.info("Answer not grounded — retrying generation")
    return "generation"


# ── Retry-count increment ──────────────────────────────────────────────────────

def increment_retry_node(state: RAGState) -> dict:
    """
    Lightweight node that increments retry_count before re-entering
    query_analysis on the retry path. Needed because conditional edges
    can't mutate state directly.
    """
    return {"retry_count": state.get("retry_count", 0) + 1}


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph(llm, generation_llm, vectorstore: VectorStore, settings: Settings):
    """
    Construct and compile the LangGraph StateGraph.

    Separate llm and generation_llm allow using a cheaper model for
    grading/analysis and a higher-quality model for final answer generation.
    """
    top_k = settings.retrieval_top_k
    max_retries = settings.max_retries
    tavily_key = settings.tavily_api_key

    # ── Bind arguments to node functions via functools.partial ─────────────────
    # This lets us pass node callables to add_node without factory classes.

    _query_analysis = functools.partial(query_analysis_node, llm=llm)
    _retrieval = functools.partial(retrieval_node, vectorstore=vectorstore, top_k=top_k)
    _grading = functools.partial(document_grading_node, llm=llm)
    _generation = functools.partial(generation_node, llm=generation_llm)
    _hallucination = functools.partial(hallucination_check_node, llm=llm)
    _web_search = functools.partial(web_search_node, tavily_api_key=tavily_key)
    _route_grading = functools.partial(route_after_grading, max_retries=max_retries)

    # ── Build the graph ────────────────────────────────────────────────────────
    graph = StateGraph(RAGState)

    graph.add_node("query_analysis", _query_analysis)
    graph.add_node("retrieval", _retrieval)
    graph.add_node("document_grading", _grading)
    graph.add_node("increment_retry", increment_retry_node)
    graph.add_node("web_search", _web_search)
    graph.add_node("generation", _generation)
    graph.add_node("hallucination_check", _hallucination)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "document_grading")

    # Conditional: after grading decide whether to generate, retry, or web-search
    graph.add_conditional_edges(
        "document_grading",
        _route_grading,
        {
            "generation": "generation",
            "query_analysis": "increment_retry",   # bump counter, then rewrite
            "web_search": "web_search",
        },
    )

    # Retry loop: increment → query_analysis → retrieval → grading
    graph.add_edge("increment_retry", "query_analysis")

    # Web search feeds directly into generation
    graph.add_edge("web_search", "generation")

    # After generation → check for hallucinations
    graph.add_edge("generation", "hallucination_check")

    # Conditional: accept answer or regenerate once
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "generation": "generation",
            "__end__": END,
        },
    )

    return graph.compile()
