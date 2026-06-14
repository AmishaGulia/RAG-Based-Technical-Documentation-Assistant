"""
Bonus Node — Web Search Fallback

Triggered when the vector store has no relevant results after max_retries
attempts. Uses the Tavily API (AI-optimised web search) to find relevant
web pages and wraps the results as LangChain Document objects so the
generation node can use them transparently.

Falls back gracefully (empty list) when TAVILY_API_KEY is not set.
"""

import logging
from langchain_core.documents import Document

from app.rag.state import RAGState

logger = logging.getLogger(__name__)


def web_search_node(state: RAGState, tavily_api_key: str) -> dict:
    """
    Search the web for the (rewritten) question using Tavily.
    Returns results as Documents with metadata marking them as web-sourced.
    """
    query = state.get("rewritten_query") or state["question"]

    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — skipping web search")
        return {"web_search_results": [], "web_search_used": False}

    try:
        # Import here to avoid hard dependency when Tavily is not configured
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=tavily_api_key)
        logger.info("Web search for: %s", query)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_raw_content=False,
        )

        results: list[Document] = []
        for item in response.get("results", []):
            content = item.get("content", "") or item.get("snippet", "")
            url = item.get("url", "web")
            title = item.get("title", "")
            if content:
                results.append(
                    Document(
                        page_content=f"{title}\n\n{content}",
                        metadata={"source": url, "chunk_index": 0, "from_web": True},
                    )
                )

        logger.info("Web search returned %d results", len(results))
        return {"web_search_results": results, "web_search_used": len(results) > 0}

    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return {"web_search_results": [], "web_search_used": False}
