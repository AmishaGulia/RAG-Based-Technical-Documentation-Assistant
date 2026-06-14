"""
Embedding model factory.

Mirrors the _build_llm() pattern in app/main.py: EMBEDDING_PROVIDER selects
the backend; EMBEDDING_MODEL sets the specific model within that backend.

Supported providers:
  huggingface  — local sentence-transformers (default, no API key required)
  openai       — text-embedding-3-small, text-embedding-ada-002, etc.
  cohere       — embed-english-v3.0, embed-multilingual-v3.0, etc.
  google       — models/text-embedding-004, etc.

Provider packages for non-default choices:
  openai   → already in requirements (langchain-openai)
  cohere   → pip install langchain-cohere
  google   → pip install langchain-google-genai

IMPORTANT: changing EMBEDDING_PROVIDER or EMBEDDING_MODEL after documents have
been ingested requires a full re-ingest (different models produce incompatible
vector spaces). Run: python scripts/ingest_docs.py --reset
"""

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Sensible default model for each provider
_DEFAULT_MODELS: dict[str, str] = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai":      "text-embedding-3-small",
    "cohere":      "embed-english-v3.0",
    "google":      "models/text-embedding-004",
}


def build_embeddings(settings) -> Embeddings:
    """
    Instantiate an embedding model for the provider named in settings.embedding_provider.
    Raises ValueError for unknown providers; raises ImportError for missing packages.
    """
    provider = settings.embedding_provider.lower()
    model = settings.embedding_model or _DEFAULT_MODELS.get(provider, "")

    logger.info("Building embeddings — provider: %s, model: %s", provider, model)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key,
        )

    if provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install the Cohere integration: pip install langchain-cohere"
            ) from exc
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY must be set when EMBEDDING_PROVIDER=cohere")
        return CohereEmbeddings(
            model=model,
            cohere_api_key=settings.cohere_api_key,
        )

    if provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install the Google integration: pip install langchain-google-genai"
            ) from exc
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY must be set when EMBEDDING_PROVIDER=google")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=settings.google_api_key,
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
        "Choose one of: huggingface | openai | cohere | google"
    )
