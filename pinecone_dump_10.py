"""
Dump businesses from PostgreSQL to Pinecone, one business at a time:
fetch → chunk → embed → upsert.

Only rows with business_listings.ai_status = 'ai_done', pinecone_dump_status = false,
and existing seo_data are queued.
pinecone_dump_log is kept for audit/debug history, but business_listings.pinecone_dump_status
is the source of truth for sync state.

Optional cap for testing: PINECONE_DUMP_LIMIT in .env.

Usage:
  Set in .env: OPENAI_API_KEY, DATABASE_URL, PINECONE_API_KEY, PINECONE_HOST
  Optional: PINECONE_DUMP_LIMIT=10  (omit for all rows)
  python pinecone_dump_10.py

Pinecone index "test" must exist with dimension 1024 (text-embedding-3-large).
"""

import os
import uuid
from typing import Any

from dotenv import load_dotenv
import psycopg2
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from db import save_pinecone_dump_log

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST", "test-1-mrmkcuf.svc.aped-4627-b74a.pinecone.io")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test-1")
EMBEDDING_DIMENSION = 1024
# If set (integer string), only fetch that many IDs — useful for smoke tests.
_dump_limit_raw = os.getenv("PINECONE_DUMP_LIMIT", "").strip()
DUMP_LIMIT: int | None = int(_dump_limit_raw) if _dump_limit_raw.isdigit() else None


# ---------------------------------------------------------------------------
# STEP 1 — Fetch business IDs (with AI data)
# ---------------------------------------------------------------------------

def get_business_ids() -> list[int]:
    """
    Business IDs ready for Pinecone sync, ordered.
    Source of truth is business_listings.pinecone_dump_status, while seo_data is only an
    eligibility check so incomplete businesses are skipped.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    base = """
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
        cursor.execute(base + " LIMIT %s", (DUMP_LIMIT,))
    else:
        cursor.execute(base)
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return ids


def mark_pinecone_dump_status(business_id: int, status: bool) -> None:
    """Persist the business-level Pinecone sync flag after a successful upsert."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE business_listings
        SET pinecone_dump_status = %s
        WHERE id = %s
        """,
        (status, business_id),
    )
    conn.commit()
    cursor.close()
    conn.close()


def fetch_full_business_data(business_id: int) -> dict[str, Any] | None:
    """Fetch one business with seo, faqs, reviews, highlights, ctas. Returns dicts, not tuples."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Business (explicit columns for dict)
    cursor.execute("""
        SELECT id, business_name, city, business_address, message, website_url, mobile_number
        FROM business_listings
        WHERE id = %s
    """, (business_id,))
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return None

    business = {
        "id": row[0],
        "name": row[1],
        "city": row[2],
        "business_address": row[3] or "",
        "description": row[4] or "",
        "website_url": row[5] or "",
        "mobile_number": row[6] or "",
    }

    # SEO
    cursor.execute("""
        SELECT meta_title, meta_description, focus_keywords, url_slug, h1, image_alt_text
        FROM seo_data WHERE business_id = %s
    """, (business_id,))
    seo_row = cursor.fetchone()
    seo = {
        "meta_title": (seo_row[0] or ""),
        "meta_description": (seo_row[1] or ""),
        "focus_keywords": (seo_row[2] or ""),
        "url_slug": (seo_row[3] or ""),
        "h1": (seo_row[4] or ""),
        "image_alt_text": (seo_row[5] or ""),
    } if seo_row else {}

    # FAQs
    cursor.execute("""
        SELECT language, question, answer FROM faqs WHERE business_id = %s ORDER BY id
    """, (business_id,))
    faqs = {"english": [], "roman_urdu": []}
    for lang, q, a in cursor.fetchall():
        if lang in faqs:
            faqs[lang].append({"question": q or "", "answer": a or ""})

    # Reviews
    cursor.execute("""
        SELECT name, review, language, review_score, review_date
        FROM reviews WHERE business_id = %s ORDER BY review_date DESC
    """, (business_id,))
    reviews = [
        {"name": r[0], "review": r[1], "language": r[2], "score": r[3], "date": r[4]}
        for r in cursor.fetchall()
    ]

    # Highlights
    cursor.execute("""
        SELECT business_heading, products_or_services, business_highlights, competency_highlights
        FROM business_highlights WHERE business_id = %s
    """, (business_id,))
    h_row = cursor.fetchone()
    highlights = {
        "business_heading": (h_row[0] or ""),
        "products_or_services": (h_row[1] or ""),
        "business_highlights": (h_row[2] or ""),
        "competency_highlights": (h_row[3] or ""),
    } if h_row else {}

    # CTAs
    cursor.execute("""SELECT cta FROM ctas WHERE business_id = %s""", (business_id,))
    ctas = [row[0] or "" for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return {
        "business": business,
        "seo": seo,
        "faqs": faqs,
        "reviews": reviews,
        "highlights": highlights,
        "ctas": ctas,
    }


# ---------------------------------------------------------------------------
# STEP 2 — Semantic chunking (same rules as legacy one-shot dump script)
# ---------------------------------------------------------------------------

def create_chunks(business_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one business's data into semantic chunks.
    Each chunk: { "text": str, "metadata": { business_id, type, language?, field?, ... } }
    Same chunk boundaries/types as the original pinecone_dump script (description, seo, faq,
    review, highlight, cta).
    """
    bid = business_data["business"]["id"]
    chunks = []

    # 1) Business description — always one chunk per business (name, address/city, description)
    name = (business_data["business"].get("name") or "").strip()
    city = (business_data["business"].get("city") or "").strip()
    address = (business_data["business"].get("business_address") or "").strip()
    location = ", ".join(part for part in [address, city] if part)
    desc = (business_data["business"].get("description") or "").strip()
    text = (
        f"Business: {name}. Location: {location}. Description: {desc}"
        if (name or location or desc)
        else f"Business ID: {bid}"
    )
    chunks.append({
        "text": text.strip(),
        "metadata": {"business_id": bid, "type": "description"},
    })

    # Separate address chunk so location-specific searches can target address metadata directly.
    if location:
        chunks.append({
            "text": f"Business address: {location}",
            "metadata": {
                "business_id": bid,
                "type": "address",
                "business_address": location,
                "city": city,
            },
        })

    # 2) SEO — one chunk when any of title / meta / keywords / h1
    seo = business_data.get("seo") or {}
    meta_title = (seo.get("meta_title") or "").strip()
    meta_desc = (seo.get("meta_description") or "").strip()
    keywords = (seo.get("focus_keywords") or "").strip()
    h1 = (seo.get("h1") or "").strip()
    if any([meta_title, meta_desc, keywords, h1]):
        text = f"Title: {meta_title}. Description: {meta_desc}. Keywords: {keywords}. H1: {h1}"
        chunks.append({
            "text": text.strip(),
            "metadata": {"business_id": bid, "type": "seo"},
        })

    # 3) FAQs — one chunk per Q&A with language
    for lang in ["english", "roman_urdu"]:
        for faq in business_data.get("faqs", {}).get(lang) or []:
            q = (faq.get("question") or "").strip()
            a = (faq.get("answer") or "").strip()
            text = f"Q: {q} A: {a}" if (q or a) else ""
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {"business_id": bid, "type": "faq", "language": lang},
                })

    # 4) Reviews — one chunk per review with language
    for r in business_data.get("reviews") or []:
        rev = (r.get("review") or "").strip()
        lang = (r.get("language") or "").strip() or "unknown"
        name_reviewer = (r.get("name") or "").strip()
        if rev:
            text = f"Review by {name_reviewer}: {rev}"
            chunks.append({
                "text": text,
                "metadata": {"business_id": bid, "type": "review", "language": lang},
            })
        elif name_reviewer:
            text = f"Review by {name_reviewer}"
            chunks.append({
                "text": text,
                "metadata": {"business_id": bid, "type": "review", "language": lang},
            })

    # 5) Highlights — separate chunk per non-empty field + one combined
    h = business_data.get("highlights") or {}
    for field, label in [
        ("business_heading", "Business heading"),
        ("products_or_services", "Products or services"),
        ("business_highlights", "Business highlights"),
        ("competency_highlights", "Competency highlights"),
    ]:
        val = (h.get(field) or "").strip()
        if val:
            chunks.append({
                "text": f"{label}: {val}",
                "metadata": {"business_id": bid, "type": "highlight", "field": field},
            })
    parts = [
        h.get("business_heading"),
        h.get("products_or_services"),
        h.get("business_highlights"),
        h.get("competency_highlights"),
    ]
    combined = " ".join(p for p in parts if p).strip()
    if combined:
        chunks.append({
            "text": f"Highlights: {combined}",
            "metadata": {"business_id": bid, "type": "highlight", "field": "combined"},
        })

    # 6) CTAs — one chunk per CTA + combined when more than one
    ctas = [c.strip() for c in (business_data.get("ctas") or []) if c and c.strip()]
    for cta in ctas:
        chunks.append({
            "text": f"CTA: {cta}",
            "metadata": {"business_id": bid, "type": "cta"},
        })
    if len(ctas) > 1:
        chunks.append({
            "text": "CTAs: " + " | ".join(ctas),
            "metadata": {"business_id": bid, "type": "cta", "field": "combined"},
        })

    return chunks


# ---------------------------------------------------------------------------
# STEP 3 & 4 — Embeddings + Pinecone
# ---------------------------------------------------------------------------

def get_embeddings_model():
    """OpenAI text-embedding-3-large with dimension 1024 for Pinecone index."""
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=EMBEDDING_DIMENSION,
        openai_api_key=OPENAI_API_KEY,
    )


def store_in_pinecone(
    chunks: list[dict],
    embeddings_model: OpenAIEmbeddings,
    index,
    namespace: str | None = None,
) -> int:
    """
    Embed all chunk texts and upsert to Pinecone.
    metadata: only str, int, float, bool (Pinecone requirement).
    Returns number of vectors upserted.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    # Build Pinecone format: id, values, metadata (flat; only primitive types)
    upsert_list = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        meta = chunk["metadata"]
        # Pinecone metadata: only string, number, boolean
        safe_meta = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)

        uid = str(uuid.uuid4())
        upsert_list.append({
            "id": uid,
            "values": vec,
            "metadata": {"text": chunk["text"][:40000], **safe_meta},
        })

    # Upsert in batches of 100
    batch_size = 100
    upsert_kwargs = {"namespace": namespace} if namespace else {}
    for j in range(0, len(upsert_list), batch_size):
        batch = upsert_list[j : j + batch_size]
        index.upsert(vectors=batch, **upsert_kwargs)

    return len(upsert_list)


def delete_existing_pinecone_vectors(index, business_id: int, namespace: str | None = None) -> None:
    """Delete old vectors for this business before writing the fresh embedding set."""
    delete_kwargs = {"namespace": namespace} if namespace else {}
    index.delete(
        filter={"business_id": {"$eq": business_id}},
        **delete_kwargs,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not OPENAI_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY in .env")
    if not DATABASE_URL:
        raise SystemExit("Set DATABASE_URL in .env")
    if not PINECONE_API_KEY:
        raise SystemExit("Set PINECONE_API_KEY in .env")

    scope = f"first {DUMP_LIMIT} IDs" if DUMP_LIMIT is not None else "all IDs"
    print(f"Fetching business IDs: ai_done + pinecone_dump_status=false + seo_data, {scope}...")
    ids = get_business_ids()
    if not ids:
        print("No pending businesses found for Pinecone sync. Exiting.")
        return

    print(f"Found {len(ids)} business(es) to dump. Processing one at a time...")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    embeddings_model = get_embeddings_model()

    total_vectors = 0
    for i, bid in enumerate(ids, start=1):
        data = fetch_full_business_data(bid)
        if not data:
            print(f"  [{i}/{len(ids)}] Skip {bid}: no listing row")
            save_pinecone_dump_log(
                status="skipped_no_listing",
                business_id=bid,
                message="No row in business_listings",
                pinecone_index_name=PINECONE_INDEX_NAME,
            )
            continue
        chunks = create_chunks(data)
        if not chunks:
            print(f"  [{i}/{len(ids)}] Business {bid}: 0 chunks, skip upsert")
            save_pinecone_dump_log(
                status="skipped_no_chunks",
                business_id=bid,
                message="create_chunks returned empty",
                pinecone_index_name=PINECONE_INDEX_NAME,
            )
            continue
        try:
            # Delete first so retries/manual reprocessing replace vectors instead of duplicating them.
            delete_existing_pinecone_vectors(index, bid)
            n = store_in_pinecone(chunks, embeddings_model, index)
            # Mark the business as synced only after Pinecone upsert succeeds.
            mark_pinecone_dump_status(bid, True)
        except Exception as e:
            print(f"  [{i}/{len(ids)}] Business {bid}: FAILED — {e}")
            save_pinecone_dump_log(
                status="failure",
                business_id=bid,
                message=f"{len(chunks)} chunks before delete/embed/upsert/status update",
                error_message=str(e),
                chunks_count=len(chunks),
                pinecone_index_name=PINECONE_INDEX_NAME,
            )
            continue
        total_vectors += n
        print(f"  [{i}/{len(ids)}] Business {bid}: {len(chunks)} chunks → {n} vectors upserted")
        save_pinecone_dump_log(
            status="success",
            business_id=bid,
            message="Pinecone upsert OK",
            chunks_count=len(chunks),
            vectors_upserted=n,
            pinecone_index_name=PINECONE_INDEX_NAME,
        )

    print(
        f"Done. Upserted {total_vectors} vectors total to Pinecone index '{PINECONE_INDEX_NAME}'."
    )


def dump_business_to_pinecone(*, business_id: int, environment: str) -> dict[str, Any]:
    """
    Single-business upsert for POST /dump_business_to_pinecone.
    Requires ai_status = 'ai_done'. Uses `environment` as Pinecone namespace on delete/upsert.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured")
    if not environment or not str(environment).strip():
        raise ValueError("environment must be a non-empty string")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT ai_status FROM business_listings WHERE id = %s",
        (business_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise ValueError("No business found with this ID")
    if row[0] != "ai_done":
        raise ValueError(
            f"Business must have ai_status='ai_done' before Pinecone dump (current: {row[0]!r})"
        )

    data = fetch_full_business_data(business_id)
    if not data:
        raise ValueError("No business listing row")

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
            "environment": environment,
            "chunks": 0,
            "vectors_upserted": 0,
            "index_name": PINECONE_INDEX_NAME,
        }

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    embeddings_model = get_embeddings_model()
    ns = str(environment).strip()
    try:
        # Delete first so this API endpoint can safely resync changed business data.
        delete_existing_pinecone_vectors(index, business_id, namespace=ns)
        n = store_in_pinecone(chunks, embeddings_model, index, namespace=ns)
        # Persist sync state after the namespace upsert succeeds.
        mark_pinecone_dump_status(business_id, True)
    except Exception as e:
        save_pinecone_dump_log(
            status="failure",
            business_id=business_id,
            message=f"{len(chunks)} chunks before delete/embed/upsert/status update (API)",
            error_message=str(e),
            chunks_count=len(chunks),
            pinecone_index_name=PINECONE_INDEX_NAME,
        )
        raise
    save_pinecone_dump_log(
        status="success",
        business_id=business_id,
        message="Pinecone upsert OK (API)",
        chunks_count=len(chunks),
        vectors_upserted=n,
        pinecone_index_name=PINECONE_INDEX_NAME,
    )
    return {
        "business_id": business_id,
        "environment": environment,
        "chunks": len(chunks),
        "vectors_upserted": n,
        "index_name": PINECONE_INDEX_NAME,
    }


if __name__ == "__main__":
    main()
