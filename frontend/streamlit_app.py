"""
Bonus — Streamlit frontend for the RAG Documentation Assistant.

Provides an interactive chat interface with:
- Conversation history display
- Source citations panel
- Feedback buttons
- Session persistence within the browser tab

Run with:
    streamlit run frontend/streamlit_app.py

Requires the FastAPI backend to be running at API_BASE_URL.
"""

import os
import uuid
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Docs Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Session state initialisation ───────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str, "meta": dict|None}
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None


# ── Helper functions ───────────────────────────────────────────────────────────

def call_query(question: str, session_id: str | None) -> dict:
    """POST /query and return the full response dict."""
    payload = {"question": question, "session_id": session_id}
    resp = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_feedback(rating: str, question: str, answer: str, session_id: str) -> None:
    """POST /feedback."""
    payload = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": rating,
    }
    requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)


def call_documents() -> dict:
    """GET /documents and return the response dict."""
    resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_ingest_url(url: str) -> dict:
    """POST /ingest/urls and return the response dict."""
    payload = {"urls": [url]}
    resp = requests.post(f"{API_BASE_URL}/ingest/urls", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Layout ─────────────────────────────────────────────────────────────────────

st.title("📚 RAG Technical Documentation Assistant")
st.caption("Ask questions about FastAPI, LangChain, LangGraph, Pydantic, and ChromaDB.")

# Sidebar — corpus info and ingestion
with st.sidebar:
    st.header("Corpus")

    if st.button("Refresh corpus info"):
        try:
            info = call_documents()
            st.success(f"{info['total_chunks']} chunks | {info['unique_sources']} sources")
            with st.expander("Sources"):
                for doc in info.get("documents", []):
                    st.markdown(f"**{doc['source']}** — {doc['chunk_count']} chunks")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

    st.divider()
    st.subheader("Ingest a URL")
    url_input = st.text_input("URL", placeholder="https://fastapi.tiangolo.com/...")
    if st.button("Ingest URL") and url_input:
        with st.spinner("Fetching and indexing…"):
            try:
                result = call_ingest_url(url_input)
                st.success(result["message"])
            except Exception as e:
                st.error(str(e))

    st.divider()
    if st.button("New conversation"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id or 'not started'}`")


# ── Chat history display ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(f"Type: `{meta.get('query_type', '—')}`")
            cols[1].caption(f"Grounded: {'✅' if meta.get('is_grounded') else '⚠️'}")
            cols[2].caption(f"Web: {'🌐 yes' if meta.get('web_search_used') else 'no'}")
            if meta.get("sources"):
                with st.expander("Sources"):
                    for src in meta["sources"]:
                        st.markdown(
                            f"- **{src['source']}** (chunk {src['chunk_index']})\n"
                            f"  > {src['content_preview'][:120]}…"
                        )


# ── Input box ──────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about the documentation…")

if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            try:
                response = call_query(question, st.session_state.session_id)
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        answer = response.get("answer", "No answer generated.")
        st.markdown(answer)

        # Metadata strip
        meta = {
            "query_type": response.get("query_type"),
            "is_grounded": response.get("is_grounded"),
            "web_search_used": response.get("web_search_used"),
            "sources": response.get("sources", []),
            "retry_count": response.get("retry_count", 0),
        }
        cols = st.columns(3)
        cols[0].caption(f"Type: `{meta['query_type']}`")
        cols[1].caption(f"Grounded: {'✅' if meta['is_grounded'] else '⚠️'}")
        cols[2].caption(f"Web: {'🌐 yes' if meta['web_search_used'] else 'no'}")
        if meta["retry_count"]:
            st.caption(f"Retries: {meta['retry_count']}")

        if meta["sources"]:
            with st.expander("Sources"):
                for src in meta["sources"]:
                    st.markdown(
                        f"- **{src['source']}** (chunk {src['chunk_index']})\n"
                        f"  > {src['content_preview'][:120]}…"
                    )

    # Update state
    st.session_state.session_id = response.get("session_id", st.session_state.session_id)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.session_state.last_response = response


# ── Feedback buttons (shown after each answer) ─────────────────────────────────

if st.session_state.last_response and st.session_state.messages:
    st.divider()
    st.caption("Was this answer helpful?")
    col1, col2, _ = st.columns([1, 1, 8])

    with col1:
        if st.button("👍 Yes"):
            last = st.session_state.last_response
            # Find the last user question
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            if user_msgs:
                call_feedback(
                    "thumbs_up",
                    user_msgs[-1]["content"],
                    last.get("answer", ""),
                    st.session_state.session_id or "",
                )
            st.toast("Thanks for your feedback! 🙏")

    with col2:
        if st.button("👎 No"):
            last = st.session_state.last_response
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            if user_msgs:
                call_feedback(
                    "thumbs_down",
                    user_msgs[-1]["content"],
                    last.get("answer", ""),
                    st.session_state.session_id or "",
                )
            st.toast("Thanks — we'll use this to improve! 🙏")
