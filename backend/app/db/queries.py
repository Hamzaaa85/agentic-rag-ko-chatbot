"""
Pure SQL layer — only talks to Postgres.

- No OpenAI / Pinecone imports here.
- tools/* call these functions; change SQL here without touching LangGraph.
"""

from typing import Any

from psycopg2.extensions import cursor as Cursor

from backend.app.schemas.search import BusinessListItem, BusinessSearchFilters

# ---------------------------------------------------------------------------
# SQL constants (single place to edit table/column lists)
# ---------------------------------------------------------------------------

SQL_LISTING_BY_ID = """
    SELECT
        b.id,
        b.full_name,
        b.business_name,
        b.mobile_number,
        b.whatsapp_number,
        b.email,
        b.has_website,
        b.preferred_language,
        b.business_address,
        b.city,
        b.category_id,
        c.name AS category_name,
        b.sub_category_id,
        sc.name AS sub_category_name,
        b.package_status::text AS package_status,
        b.message,
        b.facebook_social_link,
        b.instagram_social_link,
        b.website_url,
        b.business_model::text AS business_model,
        b.ai_status::text AS ai_status,
        b.logo,
        b.slug,
        b.source::text AS source
    FROM business_listings b
    LEFT JOIN categories c ON c.id = b.category_id
    LEFT JOIN categories sc ON sc.id = b.sub_category_id
    WHERE b.id = %s
"""

SQL_SEO_BY_BUSINESS = """
    SELECT meta_title, meta_description, focus_keywords, url_slug, h1, image_alt_text
    FROM seo_data
    WHERE business_id = %s
    ORDER BY id
"""

SQL_HIGHLIGHTS_BY_BUSINESS = """
    SELECT
        business_heading,
        products_or_services,
        business_highlights,
        competency_highlights
    FROM business_highlights
    WHERE business_id = %s
    ORDER BY id
"""

SQL_PACKAGE_BY_BUSINESS = """
    SELECT
        business_competency_text,
        core_values_section_heading,
        core_values_items,
        deals_with_heading,
        deals_with,
        help_with_heading,
        help_with,
        second_competency_heading,
        second_competency_value
    FROM business_package_content
    WHERE business_id = %s
"""

SQL_FAQS_BY_BUSINESS = """
    SELECT language, question, answer
    FROM faqs
    WHERE business_id = %s
    ORDER BY language NULLS LAST, id
"""

SQL_REVIEWS_BY_BUSINESS = """
    SELECT name, review, language, review_score, review_date, review_type::text AS review_type
    FROM reviews
    WHERE business_id = %s
    ORDER BY review_score DESC NULLS LAST, review_date DESC NULLS LAST, id
    LIMIT 20
"""

SQL_CTAS_BY_BUSINESS = """
    SELECT cta, platform
    FROM ctas
    WHERE business_id = %s
    ORDER BY id
"""

SQL_SEARCH_SELECT = """
    SELECT
        b.id,
        b.business_name,
        b.city,
        b.slug,
        c.name AS category_name,
        sc.name AS sub_category_name,
        b.package_status::text AS package_status,
        b.has_website
    FROM business_listings b
    LEFT JOIN categories c ON c.id = b.category_id
    LEFT JOIN categories sc ON sc.id = b.sub_category_id
"""


def row_to_dict(row: Any) -> dict[str, Any] | None:
    """Convert RealDictRow to plain dict (None if row missing)."""
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Step 1 — full business bundle
# ---------------------------------------------------------------------------


def fetch_full_business_bundle(cur: Cursor, business_id: int) -> dict[str, Any] | None:
    """
    Load one business + linked tables (same shape as pinecone_dump fetch).

    Returns None if business_listings row does not exist.
    """
    cur.execute(SQL_LISTING_BY_ID, (business_id,))
    listing = cur.fetchone()
    if not listing:
        return None

    cur.execute(SQL_SEO_BY_BUSINESS, (business_id,))
    seo_rows = cur.fetchall()

    cur.execute(SQL_HIGHLIGHTS_BY_BUSINESS, (business_id,))
    highlight_rows = cur.fetchall()

    cur.execute(SQL_PACKAGE_BY_BUSINESS, (business_id,))
    package_row = cur.fetchone()

    cur.execute(SQL_FAQS_BY_BUSINESS, (business_id,))
    faq_rows = cur.fetchall()

    cur.execute(SQL_REVIEWS_BY_BUSINESS, (business_id,))
    review_rows = cur.fetchall()

    cur.execute(SQL_CTAS_BY_BUSINESS, (business_id,))
    cta_rows = cur.fetchall()

    return {
        "business": row_to_dict(listing),
        "seo_data": [row_to_dict(r) for r in seo_rows],
        "highlights": [row_to_dict(r) for r in highlight_rows],
        "package_content": row_to_dict(package_row) or {},
        "faqs": [row_to_dict(r) for r in faq_rows],
        "reviews": [row_to_dict(r) for r in review_rows],
        "ctas": [row_to_dict(r) for r in cta_rows],
    }


def fetch_business_bundles_by_ids(cur: Cursor, business_ids: list[int]) -> list[dict[str, Any]]:
    """Many ids — preserves order, skips missing ids."""
    results: list[dict[str, Any]] = []
    for business_id in business_ids:
        bundle = fetch_full_business_bundle(cur, business_id)
        if bundle:
            results.append(bundle)
    return results


# ---------------------------------------------------------------------------
# Step 2 — structured search
# ---------------------------------------------------------------------------


def _build_search_where(filters: BusinessSearchFilters) -> tuple[str, list[Any]]:
    """
    Build WHERE clause + params from filters.

    Always uses %s placeholders — never interpolate user strings into SQL.
    """
    parts: list[str] = ["WHERE 1=1"]
    params: list[Any] = []

    if filters.exclude_test_data:
        parts.append("AND COALESCE(b.is_test_data, false) = false")

    if filters.ai_status:
        parts.append("AND b.ai_status::text = %s")
        params.append(filters.ai_status)

    if filters.city:
        parts.append("AND b.city ILIKE %s")
        params.append(f"%{filters.city.strip()}%")

    if filters.category_id is not None:
        parts.append("AND b.category_id = %s")
        params.append(filters.category_id)

    if filters.category_name:
        parts.append("AND c.name ILIKE %s")
        params.append(f"%{filters.category_name.strip()}%")

    if filters.sub_category_id is not None:
        parts.append("AND b.sub_category_id = %s")
        params.append(filters.sub_category_id)

    if filters.sub_category_name:
        parts.append("AND sc.name ILIKE %s")
        params.append(f"%{filters.sub_category_name.strip()}%")

    if filters.has_website is True:
        parts.append("AND COALESCE(b.has_website, false) = true")
    elif filters.has_website is False:
        parts.append("AND COALESCE(b.has_website, false) = false")

    if filters.package_status:
        parts.append("AND b.package_status::text = %s")
        params.append(filters.package_status)

    if filters.has_instagram is True:
        parts.append(
            "AND b.instagram_social_link IS NOT NULL AND TRIM(b.instagram_social_link) <> ''"
        )
    elif filters.has_instagram is False:
        parts.append(
            "AND (b.instagram_social_link IS NULL OR TRIM(b.instagram_social_link) = '')"
        )

    if filters.has_facebook is True:
        parts.append(
            "AND b.facebook_social_link IS NOT NULL AND TRIM(b.facebook_social_link) <> ''"
        )
    elif filters.has_facebook is False:
        parts.append(
            "AND (b.facebook_social_link IS NULL OR TRIM(b.facebook_social_link) = '')"
        )

    return " ".join(parts), params


def search_business_listings(
    cur: Cursor,
    filters: BusinessSearchFilters,
    limit: int = 10,
) -> list[BusinessListItem]:
    """Exact/filter search — returns light rows for merge with Pinecone."""
    where_sql, params = _build_search_where(filters)
    sql = f"{SQL_SEARCH_SELECT} {where_sql} ORDER BY b.id LIMIT %s"
    params.append(limit)

    cur.execute(sql, params)
    rows = cur.fetchall()
    return [BusinessListItem.model_validate(dict(row)) for row in rows]
