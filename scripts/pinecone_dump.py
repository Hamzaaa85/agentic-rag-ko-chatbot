"""
Dump AI-ready businesses from PostgreSQL to Pinecone, one business at a time.

Pipeline:
  fetch pending IDs -> fetch full linked business data -> create semantic chunks
  -> embed -> delete old vectors for that business -> upsert fresh vectors
  -> mark business_listings.pinecone_dump_status = true.

Eligibility:
  - business_listings.ai_status = 'ai_done'
  - business_listings.pinecone_dump_status = false
  - seo_data row exists for the business

Environment variables:
  Required:
    OPENAI_API_KEY
    DATABASE_URL
    PINECONE_API_KEY
    PINECONE_INDEX_NAME

  Optional:
    PINECONE_HOST=...  (if omitted, resolved from Pinecone after index exists/is created)
    PINECONE_CLOUD=aws
    PINECONE_REGION=us-east-1
    PINECONE_NAMESPACE=production
    PINECONE_DUMP_LIMIT=10
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_DIMENSION=1024

Usage (from project root):
  python scripts/pinecone_dump.py

Example vector IDs for one business:
  business__77-core-profile, business__77-seo, business__77-contact, business__77-reviews

Important:
  Pinecone is only used for semantic retrieval. Final chatbot answers should
  fetch fresh full business data from PostgreSQL by business_id.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = (os.getenv("PINECONE_HOST") or "").strip() or None
PINECONE_INDEX_NAME = (os.getenv("PINECONE_INDEX_NAME") or "").strip() or "test-1"
PINECONE_CLOUD = (os.getenv("PINECONE_CLOUD") or "aws").strip()
PINECONE_REGION = (os.getenv("PINECONE_REGION") or "us-east-1").strip()
PINECONE_NAMESPACE = (os.getenv("PINECONE_NAMESPACE") or "").strip() or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

_cached_pinecone_index = None

_dump_limit_raw = os.getenv("PINECONE_DUMP_LIMIT", "").strip()
DUMP_LIMIT: int | None = int(_dump_limit_raw) if _dump_limit_raw.isdigit() else None


def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL in .env")
    return psycopg2.connect(DATABASE_URL)


def clean_text(value: Any) -> str:
    """Convert values to compact text suitable for embedding."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (datetime, date)):
        text = value.isoformat()
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def has_value(value: Any) -> bool:
    return bool(clean_text(value))


def is_placeholder(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"", "none", "null", "n/a", "na"}


def join_sentences(*parts: str) -> str:
    """Build one readable paragraph for embedding (no label-per-line noise)."""
    sentences: list[str] = []
    for part in parts:
        text = clean_text(part)
        if not text or is_placeholder(text):
            continue
        sentences.append(text.rstrip("."))
    if not sentences:
        return ""
    return ". ".join(sentences) + "."


def build_vector_id(
    business_id: int,
    chunk_type: str,
    *,
    sequence: int = 0,
    language: str | None = None,
) -> str:
    """
    Readable Pinecone vector IDs, e.g. business__77-core-profile, business__77-faqs-english.
    """
    prefix = f"business__{business_id}"
    slug = chunk_type.replace("_", "-")
    if chunk_type == "faqs_profile" and language:
        lang_slug = re.sub(r"[^a-z0-9]+", "-", language.lower()).strip("-") or "unknown"
        return f"{prefix}-faqs-{lang_slug}"
    if sequence > 0:
        return f"{prefix}-{slug}-{sequence + 1}"
    return f"{prefix}-{slug}"


def save_pinecone_dump_log(
    *,
    status: str,
    business_id: int | None = None,
    message: str | None = None,
    error_message: str | None = None,
    chunks_count: int | None = None,
    vectors_upserted: int | None = None,
    pinecone_index_name: str | None = None,
) -> None:
    """Audit/debug log. The sync source of truth remains business_listings.pinecone_dump_status."""
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pinecone_dump_log (
                    business_id,
                    status,
                    message,
                    error_message,
                    chunks_count,
                    vectors_upserted,
                    pinecone_index_name
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    business_id,
                    status,
                    message,
                    error_message,
                    chunks_count,
                    vectors_upserted,
                    pinecone_index_name,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_business_ids() -> list[int]:
    """
    Fetch business IDs ready for Pinecone sync.

    Source of truth for sync state is business_listings.pinecone_dump_status.
    seo_data is required so incomplete businesses are skipped.

    The script intentionally processes one ID at a time so a failure on one
    business does not stop the rest of the queue.
    """
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT b.id
                FROM business_listings b
                WHERE b.ai_status = 'ai_done'
                  AND COALESCE(b.pinecone_dump_status, false) = false
                  AND EXISTS (
                      SELECT 1
                      FROM seo_data s
                      WHERE s.business_id = b.id
                  )
                ORDER BY b.id
            """
            if DUMP_LIMIT is not None:
                cur.execute(query + " LIMIT %s", (DUMP_LIMIT,))
            else:
                cur.execute(query)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def mark_pinecone_dump_status(business_id: int, status: bool) -> None:
    """Mark a business as synced only after Pinecone upsert succeeds."""
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE business_listings
                SET pinecone_dump_status = %s
                WHERE id = %s
                """,
                (status, business_id),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_full_business_data(business_id: int) -> dict[str, Any] | None:
    """
    Fetch one business with all currently linked chatbot data.

    Included:
      business_listings, category/subcategory names, seo_data, highlights,
      business_package_content, faqs, reviews, and ctas.
    """
    conn = db_connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
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
                """,
                (business_id,),
            )
            business = cur.fetchone()
            if not business:
                return None

            cur.execute(
                """
                SELECT meta_title, meta_description, focus_keywords, url_slug, h1, image_alt_text
                FROM seo_data
                WHERE business_id = %s
                ORDER BY id
                """,
                (business_id,),
            )
            seo_data = cur.fetchall()

            cur.execute(
                """
                SELECT
                    business_heading,
                    products_or_services,
                    business_highlights,
                    competency_highlights
                FROM business_highlights
                WHERE business_id = %s
                ORDER BY id
                """,
                (business_id,),
            )
            highlights = cur.fetchall()

            cur.execute(
                """
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
                """,
                (business_id,),
            )
            package_content = cur.fetchone()

            cur.execute(
                """
                SELECT language, question, answer
                FROM faqs
                WHERE business_id = %s
                ORDER BY language NULLS LAST, id
                """,
                (business_id,),
            )
            faqs = cur.fetchall()

            cur.execute(
                """
                SELECT name, review, language, review_score, review_date, review_type::text AS review_type
                FROM reviews
                WHERE business_id = %s
                ORDER BY review_score DESC NULLS LAST, review_date DESC NULLS LAST, id
                LIMIT 20
                """,
                (business_id,),
            )
            reviews = cur.fetchall()

            cur.execute(
                """
                SELECT cta, platform
                FROM ctas
                WHERE business_id = %s
                ORDER BY id
                """,
                (business_id,),
            )
            ctas = cur.fetchall()

            return {
                "business": dict(business),
                "seo_data": [dict(row) for row in seo_data],
                "highlights": [dict(row) for row in highlights],
                "package_content": dict(package_content) if package_content else {},
                "faqs": [dict(row) for row in faqs],
                "reviews": [dict(row) for row in reviews],
                "ctas": [dict(row) for row in ctas],
            }
    finally:
        conn.close()


def base_metadata(business: dict[str, Any], chunk_type: str) -> dict[str, Any]:
    """Metadata used for filtering and mapping Pinecone hits back to Postgres."""
    return {
        "business_id": business.get("id"),
        "chunk_type": chunk_type,
        "business_name": business.get("business_name"),
        "city": business.get("city"),
        "category_id": business.get("category_id"),
        "sub_category_id": business.get("sub_category_id"),
        "category_name": business.get("category_name"),
        "sub_category_name": business.get("sub_category_name"),
        "slug": business.get("slug"),
        "package_status": business.get("package_status"),
    }


def create_chunks(business_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one full business payload into focused semantic chunks.

    Chunk types:
      - core_profile
      - seo_profile
      - highlights_profile
      - package_profile
      - faqs_profile
      - reviews_profile
      - cta_profile
      - contact_location
    """
    business = business_data["business"]
    business_id = int(business["id"])
    chunks: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}

    def add_chunk(
        chunk_type: str,
        text: str,
        extra_meta: dict[str, Any] | None = None,
        *,
        language: str | None = None,
    ) -> None:
        text = clean_text(text)
        if not text:
            return

        sequence = type_counts.get(chunk_type, 0)
        type_counts[chunk_type] = sequence + 1

        metadata = base_metadata(business, chunk_type)
        if extra_meta:
            metadata.update(extra_meta)

        vector_id = build_vector_id(
            business_id,
            chunk_type,
            sequence=sequence,
            language=language,
        )
        chunks.append({"id": vector_id, "text": text, "metadata": metadata})

    name = clean_text(business.get("business_name"))
    city = clean_text(business.get("city"))
    category = clean_text(business.get("category_name"))
    subcategory = clean_text(business.get("sub_category_name"))
    address = clean_text(business.get("business_address"))

    core_parts = [
        f"{name} is a {category or 'business'}",
        f"in {city}" if city else "",
        f"specializing in {subcategory}" if subcategory and not is_placeholder(subcategory) else "",
        f"Address: {address}" if address else "",
        f"Model: {business.get('business_model')}" if not is_placeholder(business.get("business_model")) else "",
        f"Package: {business.get('package_status')}" if not is_placeholder(business.get("package_status")) else "",
        clean_text(business.get("message")),
    ]
    add_chunk("core_profile", join_sentences(*core_parts))

    for index, seo in enumerate(business_data.get("seo_data") or []):
        seo_text = join_sentences(
            seo.get("meta_title"),
            seo.get("meta_description"),
            f"Keywords: {seo.get('focus_keywords')}" if not is_placeholder(seo.get("focus_keywords")) else "",
            seo.get("h1"),
            seo.get("image_alt_text"),
            f"Slug: {seo.get('url_slug')}" if not is_placeholder(seo.get("url_slug")) else "",
        )
        add_chunk("seo_profile", seo_text, {"section_index": index})

    for index, highlight in enumerate(business_data.get("highlights") or []):
        highlight_text = join_sentences(
            highlight.get("business_heading"),
            highlight.get("products_or_services"),
            highlight.get("business_highlights"),
            highlight.get("competency_highlights"),
        )
        add_chunk("highlights_profile", highlight_text, {"section_index": index})

    package = business_data.get("package_content") or {}
    package_text = join_sentences(
        package.get("business_competency_text"),
        package.get("core_values_section_heading"),
        package.get("core_values_items"),
        package.get("deals_with_heading"),
        package.get("deals_with"),
        package.get("help_with_heading"),
        package.get("help_with"),
        f"{package.get('second_competency_heading')} {package.get('second_competency_value')}".strip(),
    )
    if package_text:
        add_chunk("package_profile", package_text)

    faqs_by_language: dict[str, list[dict[str, Any]]] = {}
    for faq in business_data.get("faqs") or []:
        language = clean_text(faq.get("language")) or "unknown"
        faqs_by_language.setdefault(language, []).append(faq)

    for language, faqs in faqs_by_language.items():
        qa_parts: list[str] = []
        for faq in faqs:
            question = clean_text(faq.get("question"))
            answer = clean_text(faq.get("answer"))
            if question and answer:
                qa_parts.append(f"{question} — {answer}")
            elif question:
                qa_parts.append(question)
            elif answer:
                qa_parts.append(answer)
        if qa_parts:
            add_chunk(
                "faqs_profile",
                join_sentences(*qa_parts),
                {"language": language},
                language=language,
            )

    review_parts: list[str] = []
    review_scores: list[float] = []
    for review in business_data.get("reviews") or []:
        body = clean_text(review.get("review"))
        reviewer = clean_text(review.get("name")) or "Customer"
        score = review.get("review_score")
        if isinstance(score, (int, float)):
            review_scores.append(float(score))
        if body:
            score_text = f" (rating {score})" if score is not None else ""
            review_parts.append(f"{reviewer}{score_text}: {body}")

    if review_parts:
        review_meta: dict[str, Any] = {"review_count": len(review_parts)}
        if review_scores:
            review_meta["avg_review_score"] = round(sum(review_scores) / len(review_scores), 2)
        add_chunk("reviews_profile", join_sentences(*review_parts), review_meta)

    cta_parts: list[str] = []
    for cta in business_data.get("ctas") or []:
        cta_text = clean_text(cta.get("cta"))
        platform = clean_text(cta.get("platform"))
        if cta_text:
            cta_parts.append(f"{platform}: {cta_text}" if platform else cta_text)
    if cta_parts:
        add_chunk("cta_profile", join_sentences(*cta_parts))

    social_channels: list[str] = []
    if has_value(business.get("instagram_social_link")):
        social_channels.append("Instagram")
    if has_value(business.get("facebook_social_link")):
        social_channels.append("Facebook")
    if has_value(business.get("whatsapp_number")):
        social_channels.append("WhatsApp")

    contact_parts = [
        f"{name} is based in {city}" if name and city else "",
        f"Address: {address}" if address else "",
        f"Website: {business.get('website_url')}" if has_value(business.get("website_url")) else "",
        f"Also on {', '.join(social_channels)}" if social_channels else "",
        "Phone and email contact available on file"
        if has_value(business.get("mobile_number")) or has_value(business.get("email"))
        else "",
    ]
    add_chunk("contact_location", join_sentences(*contact_parts))

    return chunks


def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        openai_api_key=OPENAI_API_KEY,
    )


def pinecone_safe_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Pinecone metadata supports primitive values. Drop nulls and stringify complex values."""
    safe: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = clean_text(value)
    return safe


def store_in_pinecone(
    chunks: list[dict[str, Any]],
    embeddings_model: OpenAIEmbeddings,
    index,
    namespace: str | None = None,
) -> int:
    """Embed chunks and upsert vectors to Pinecone."""
    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)

    upsert_list = []
    for chunk, vector in zip(chunks, vectors):
        metadata = pinecone_safe_metadata(chunk["metadata"])
        vector_id = chunk.get("id") or build_vector_id(
            int(metadata["business_id"]),
            str(metadata.get("chunk_type", "chunk")),
        )

        upsert_list.append(
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    **metadata,
                    "text": chunk["text"][:40000],
                },
            }
        )

    upsert_kwargs = {"namespace": namespace} if namespace else {}
    batch_size = 100
    for start in range(0, len(upsert_list), batch_size):
        index.upsert(vectors=upsert_list[start : start + batch_size], **upsert_kwargs)

    return len(upsert_list)


def normalize_pinecone_host(host: str) -> str:
    """Pinecone Index(host=...) expects a hostname, not a full URL."""
    return host.replace("https://", "").replace("http://", "").strip().strip("/")


def _is_missing_namespace_error(exc: Exception) -> bool:
    """True when delete runs before any vectors exist in the index/namespace."""
    message = str(exc).lower()
    return "namespace not found" in message or "404" in message


def delete_existing_pinecone_vectors(index, business_id: int, namespace: str | None = None) -> None:
    """
    Delete existing vectors for one business before writing fresh chunks.

    On a brand-new empty index Pinecone may return 404 Namespace not found; that is safe
    to ignore because there is nothing to delete yet.
    """
    delete_kwargs = {"namespace": namespace} if namespace else {}
    try:
        index.delete(filter={"business_id": {"$eq": business_id}}, **delete_kwargs)
    except Exception as exc:
        if _is_missing_namespace_error(exc):
            return
        raise


def _index_is_ready(desc: Any) -> bool:
    status = getattr(desc, "status", None)
    if status is None:
        return False
    if hasattr(status, "ready"):
        return bool(status.ready)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return str(status).lower() in {"ready", "true"}


def wait_for_index_ready(pc: Pinecone, index_name: str, timeout_seconds: int = 180) -> None:
    """Wait until Pinecone reports the index is ready for upserts."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        if _index_is_ready(pc.describe_index(index_name)):
            return
        time.sleep(2)
    raise TimeoutError(f"Pinecone index '{index_name}' was not ready within {timeout_seconds}s")


def ensure_pinecone_index(pc: Pinecone) -> str:
    """
    Ensure the configured Pinecone index exists, creating it if missing.

    Returns the index host used by pc.Index(host=...).
    """
    index_name = PINECONE_INDEX_NAME
    if not pc.has_index(index_name):
        print(
            f"Pinecone index '{index_name}' not found. Creating serverless index "
            f"(dimension={EMBEDDING_DIMENSION}, metric=cosine, cloud={PINECONE_CLOUD}, "
            f"region={PINECONE_REGION})..."
        )
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        wait_for_index_ready(pc, index_name)
        print(f"Pinecone index '{index_name}' created.")

    if PINECONE_HOST:
        return normalize_pinecone_host(PINECONE_HOST)

    host = pc.describe_index(index_name).host
    if not host:
        raise RuntimeError(f"Could not resolve Pinecone host for index '{index_name}'")
    host = normalize_pinecone_host(host)
    print(f"Resolved Pinecone host for '{index_name}': {host}")
    return host


def get_pinecone_index():
    """Return a cached Pinecone Index client, creating the index on first use if needed."""
    global _cached_pinecone_index
    if _cached_pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        host = ensure_pinecone_index(pc)
        _cached_pinecone_index = pc.Index(host=host)
    return _cached_pinecone_index


def validate_environment() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY in .env")
    if not DATABASE_URL:
        raise SystemExit("Set DATABASE_URL in .env")
    if not PINECONE_API_KEY:
        raise SystemExit("Set PINECONE_API_KEY in .env")
    if not PINECONE_INDEX_NAME:
        raise SystemExit("Set PINECONE_INDEX_NAME in .env")


def dump_business_to_pinecone(*, business_id: int, environment: str | None = None) -> dict[str, Any]:
    """
    Upsert one business to Pinecone.

    This is useful for an API endpoint or manual re-sync. It keeps the same
    ai_done gate as the batch job.
    """
    validate_environment()
    namespace = (environment or PINECONE_NAMESPACE or "").strip() or None

    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ai_status::text FROM business_listings WHERE id = %s", (business_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(f"No business found with id={business_id}")
    if row[0] != "ai_done":
        raise ValueError(f"Business must have ai_status='ai_done' before dump. Current: {row[0]!r}")

    data = fetch_full_business_data(business_id)
    if not data:
        raise ValueError(f"No business listing row found for id={business_id}")

    chunks = create_chunks(data)
    if not chunks:
        save_pinecone_dump_log(
            status="skipped_no_chunks",
            business_id=business_id,
            message="create_chunks returned empty",
            pinecone_index_name=PINECONE_INDEX_NAME,
        )
        return {
            "business_id": business_id,
            "namespace": namespace,
            "chunks": 0,
            "vectors_upserted": 0,
            "index_name": PINECONE_INDEX_NAME,
        }

    index = get_pinecone_index()
    embeddings_model = get_embeddings_model()

    try:
        delete_existing_pinecone_vectors(index, business_id, namespace=namespace)
        vectors_upserted = store_in_pinecone(chunks, embeddings_model, index, namespace=namespace)
        mark_pinecone_dump_status(business_id, True)
    except Exception as exc:
        save_pinecone_dump_log(
            status="failure",
            business_id=business_id,
            message=f"{len(chunks)} chunks before delete/embed/upsert/status update",
            error_message=str(exc),
            chunks_count=len(chunks),
            pinecone_index_name=PINECONE_INDEX_NAME,
        )
        raise

    save_pinecone_dump_log(
        status="success",
        business_id=business_id,
        message="Pinecone upsert OK",
        chunks_count=len(chunks),
        vectors_upserted=vectors_upserted,
        pinecone_index_name=PINECONE_INDEX_NAME,
    )
    return {
        "business_id": business_id,
        "namespace": namespace,
        "chunks": len(chunks),
        "vectors_upserted": vectors_upserted,
        "index_name": PINECONE_INDEX_NAME,
    }


def main() -> None:
    validate_environment()
    get_pinecone_index()

    scope = f"first {DUMP_LIMIT} IDs" if DUMP_LIMIT is not None else "all IDs"
    namespace_label = PINECONE_NAMESPACE or "default"
    print(
        f"Fetching business IDs: ai_done + pinecone_dump_status=false + seo_data, {scope}..."
    )
    print(f"Pinecone index: {PINECONE_INDEX_NAME}, namespace: {namespace_label}")

    business_ids = get_business_ids()
    if not business_ids:
        print("No pending businesses found for Pinecone sync. Exiting.")
        return

    print(f"Found {len(business_ids)} business(es). Processing one at a time...")

    total_vectors = 0
    for index, business_id in enumerate(business_ids, start=1):
        try:
            result = dump_business_to_pinecone(
                business_id=business_id,
                environment=PINECONE_NAMESPACE,
            )
        except Exception as exc:
            print(f"  [{index}/{len(business_ids)}] Business {business_id}: FAILED - {exc}")
            continue

        total_vectors += int(result["vectors_upserted"])
        print(
            f"  [{index}/{len(business_ids)}] Business {business_id}: "
            f"{result['chunks']} chunks -> {result['vectors_upserted']} vectors upserted"
        )

    print(
        f"Done. Upserted {total_vectors} vectors total to Pinecone index "
        f"'{PINECONE_INDEX_NAME}'."
    )


if __name__ == "__main__":
    main()
