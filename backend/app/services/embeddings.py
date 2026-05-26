"""
Embedding client — vectorize user queries for Pinecone search.

Indexing stays in scripts/pinecone_dump.py; this file is query-time only.
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from backend.app.config import get_settings


@lru_cache
def get_embeddings_model() -> OpenAIEmbeddings:
    """Shared OpenAI embeddings client (same model + dimension as Pinecone index)."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
        openai_api_key=settings.openai_api_key,
    )


def embed_query(text: str) -> list[float]:
    """Convert one search sentence into a vector for index.query()."""
    model = get_embeddings_model()
    return model.embed_query(text)
