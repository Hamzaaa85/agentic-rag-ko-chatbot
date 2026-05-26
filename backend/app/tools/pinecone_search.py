"""
Step 3 tool — semantic search on Pinecone.

Returns business_id + chunk hints — full data from business_details tool.
"""

from typing import Any

from backend.app.config import get_settings
from backend.app.schemas.search import PineconeMatch
from backend.app.services.embeddings import embed_query
from backend.app.services.pinecone_client import get_pinecone_index


def _build_metadata_filter(
    city: str | None = None,
    category_id: int | None = None,
    chunk_type: str | None = None,
) -> dict[str, Any] | None:
    """
    Optional Pinecone metadata filter (AND). Independent from Postgres filters.

    City is normalized to Title Case because Pinecone metadata filters are
    case-sensitive and business data is stored in Title Case (e.g. 'Karachi').
    """
    conditions: list[dict[str, Any]] = []

    if city:
        normalized_city = city.strip().title()
        conditions.append({"city": {"$eq": normalized_city}})

    if category_id is not None:
        conditions.append({"category_id": {"$eq": category_id}})

    if chunk_type:
        conditions.append({"chunk_type": {"$eq": chunk_type}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _extract_matches_from_query_response(response: Any) -> list[Any]:
    """
    Pinecone SDK v5+ returns QueryResponse with .matches (not a plain dict).

    Supports .matches, .to_dict(), or legacy dict for version compatibility.
    """
    if hasattr(response, "matches"):
        return list(response.matches or [])

    if hasattr(response, "to_dict"):
        data = response.to_dict()
        if isinstance(data, dict):
            return list(data.get("matches") or [])

    if isinstance(response, dict):
        return list(response.get("matches") or [])

    return []


def _extract_match_fields(match: Any) -> tuple[dict[str, Any], float]:
    """Read metadata dict and score from ScoredVector or legacy dict."""
    if hasattr(match, "metadata"):
        meta = match.metadata if isinstance(match.metadata, dict) else {}
        score = float(getattr(match, "score", None) or 0.0)
        return meta, score

    if isinstance(match, dict):
        meta = match.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        score = float(match.get("score") or 0.0)
        return meta, score

    return {}, 0.0


def search_pinecone(
    query: str,
    *,
    top_k: int = 10,
    city: str | None = None,
    category_id: int | None = None,
    chunk_type: str | None = None,
) -> list[PineconeMatch]:
    """
    Embed query -> Pinecone query -> matches.

    Duplicate business_ids possible (multiple chunks); dedupe in graph merge.
    """
    if not query or not query.strip():
        return []

    settings = get_settings()
    index = get_pinecone_index()
    vector = embed_query(query.strip())

    query_kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    metadata_filter = _build_metadata_filter(
        city=city,
        category_id=category_id,
        chunk_type=chunk_type,
    )
    if metadata_filter:
        query_kwargs["filter"] = metadata_filter

    if settings.pinecone_namespace:
        query_kwargs["namespace"] = settings.pinecone_namespace

    response = index.query(**query_kwargs)

    matches: list[PineconeMatch] = []
    for match in _extract_matches_from_query_response(response):
        meta, score = _extract_match_fields(match)
        business_id = meta.get("business_id")
        if business_id is None:
            continue

        matches.append(
            PineconeMatch(
                business_id=int(business_id),
                chunk_type=meta.get("chunk_type"),
                score=score,
                text_snippet=(meta.get("text") or "")[:500] or None,
                business_name=meta.get("business_name"),
                city=meta.get("city"),
            )
        )

    return matches


def search_pinecone_business_ids(
    query: str,
    *,
    top_k: int = 10,
    city: str | None = None,
    category_id: int | None = None,
    chunk_type: str | None = None,
) -> list[int]:
    """Unique business_ids in score order (first chunk per id wins)."""
    seen: set[int] = set()
    ordered: list[int] = []

    for hit in search_pinecone(
        query,
        top_k=top_k,
        city=city,
        category_id=category_id,
        chunk_type=chunk_type,
    ):
        if hit.business_id in seen:
            continue
        seen.add(hit.business_id)
        ordered.append(hit.business_id)

    return ordered
