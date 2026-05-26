"""
Pinecone index handle — connection only.

Semantic search logic lives in tools/pinecone_search.py.
"""

from functools import lru_cache

from pinecone import Pinecone

from backend.app.config import get_settings


def normalize_pinecone_host(host: str) -> str:
    """Pinecone Index(host=...) needs hostname without https://."""
    return host.replace("https://", "").replace("http://", "").strip().strip("/")


def resolve_pinecone_host(pc: Pinecone, index_name: str, configured_host: str | None) -> str:
    """Use .env PINECONE_HOST if set, else describe_index from API."""
    if configured_host:
        return normalize_pinecone_host(configured_host)

    host = pc.describe_index(index_name).host
    if not host:
        raise RuntimeError(f"Could not resolve Pinecone host for index '{index_name}'")
    return normalize_pinecone_host(host)


@lru_cache
def get_pinecone_index():
    """
    Cached Index client for chat/search.

    Index creation is scripts/pinecone_dump.py — not done here.
    """
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    host = resolve_pinecone_host(
        pc,
        settings.pinecone_index_name,
        settings.pinecone_host,
    )
    return pc.Index(host=host)
