"""
Step 1 tool — full business from Postgres (source of truth for answers).

Depends only on db/* — no Pinecone, no LangGraph.
"""

from typing import Any

from backend.app.db.connection import get_db_cursor
from backend.app.db import queries


def get_business_by_id(business_id: int) -> dict[str, Any] | None:
    """
    One business bundle: listing + seo + highlights + package + faqs + reviews + ctas.

    Use this for final answers — not raw Pinecone chunk text.
    """
    with get_db_cursor() as cur:
        return queries.fetch_full_business_bundle(cur, business_id)


def get_businesses_by_ids(business_ids: list[int]) -> list[dict[str, Any]]:
    """Many ids in merge order; skips ids that do not exist."""
    if not business_ids:
        return []

    with get_db_cursor() as cur:
        return queries.fetch_business_bundles_by_ids(cur, business_ids)
