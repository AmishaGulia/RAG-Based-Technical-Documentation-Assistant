"""
Application configuration loaded from environment variables / .env file.
All tuneable parameters live here so no magic strings are scattered in code.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "anthropic"           # anthropic | openai | groq
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # Model used for grading, query analysis, hallucination check (cheap/fast)
    llm_model: str = "claude-haiku-4-5-20251001"
    # Model used for final answer generation (can be upgraded for quality)
    generation_model: str = "claude-haiku-4-5-20251001"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Select provider: huggingface | openai | cohere | google
    embedding_provider: str = "huggingface"
    # Default model per provider:
    #   huggingface → sentence-transformers/all-MiniLM-L6-v2  (local, no key)
    #   openai      → text-embedding-3-small
    #   cohere      → embed-english-v3.0
    #   google      → models/text-embedding-004
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cohere_api_key: str = ""
    google_api_key: str = ""

    # ── Vector store ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "tech_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    # How many times the graph retries with a rewritten query before giving up
    max_retries: int = 2

    # ── Web search fallback (bonus) ───────────────────────────────────────────
    tavily_api_key: str = ""
    enable_web_search: bool = True

    # ── Application ───────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    feedback_dir: str = "./data/feedback"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton; avoids re-parsing .env on every call."""
    return Settings()
