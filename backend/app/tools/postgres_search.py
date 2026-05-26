"""
Step 2 tool — structured Postgres search (exact filters).

Depends on db/queries + schemas/search — no Pinecone, no LLM.
"""

from backend.app.db.connection import get_db_cursor
from backend.app.db import queries
from backend.app.schemas.search import BusinessListItem, BusinessSearchFilters


def _parse_filters(filters: BusinessSearchFilters | dict | None) -> BusinessSearchFilters:
    """Accept Pydantic model or planner dict from LangGraph."""
    if filters is None:
        return BusinessSearchFilters()
    if isinstance(filters, dict):
        return BusinessSearchFilters.model_validate(filters)
    return filters


def search_businesses(
    filters: BusinessSearchFilters | dict | None = None,
    limit: int = 10,
) -> list[BusinessListItem]:
    """
    Filter search on business_listings.

    Example filters: {"city": "Karachi", "has_website": true}
    """
    parsed = _parse_filters(filters)

    with get_db_cursor() as cur:
        return queries.search_business_listings(cur, parsed, limit=limit)


def search_business_ids(
    filters: BusinessSearchFilters | dict | None = None,
    limit: int = 10,
) -> list[int]:
    """Return only ids — for merge node in LangGraph."""
    rows = search_businesses(filters=filters, limit=limit)
    return [row.id for row in rows]
