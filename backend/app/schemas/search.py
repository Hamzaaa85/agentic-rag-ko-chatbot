"""
Search contracts — shared shapes for Postgres/Pinecone tools.

Changing filters here does not change SQL; see db/queries.py for SQL.
"""

from pydantic import BaseModel


class BusinessSearchFilters(BaseModel):
    """Structured filters the LLM planner may send (never raw SQL)."""

    city: str | None = None  # e.g. "Karachi" — partial match (ILIKE)
    category_id: int | None = None  # exact categories.id
    category_name: str | None = None  # fuzzy match on categories.name
    sub_category_id: int | None = None
    sub_category_name: str | None = None
    has_website: bool | None = None  # True = must have website flag
    package_status: str | None = None  # e.g. "Basic"
    has_instagram: bool | None = None
    has_facebook: bool | None = None
    ai_status: str | None = "ai_done"  # chatbot usually only AI-ready listings
    exclude_test_data: bool = True  # skip rows where is_test_data is true


class BusinessListItem(BaseModel):
    """Light row returned by Postgres search (not the full business bundle)."""

    id: int
    business_name: str | None = None
    city: str | None = None
    slug: str | None = None
    category_name: str | None = None
    sub_category_name: str | None = None
    package_status: str | None = None
    has_website: bool | None = None


class PineconeMatch(BaseModel):
    """One semantic hit from Pinecone (ranking hint — not the final answer source)."""

    business_id: int
    chunk_type: str | None = None
    score: float
    text_snippet: str | None = None
    business_name: str | None = None
    city: str | None = None
