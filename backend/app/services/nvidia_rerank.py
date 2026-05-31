import requests
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import get_settings

NVIDIA_RANKING_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
NVIDIA_MODEL = "nvidia/rerank-qa-mistral-4b"


def _bundle_to_passage(bundle: Dict[str, Any]) -> str:
    """
    Build one searchable passage from a business bundle.

    `businesses` items are bundles: {"business": {...}, "highlights": [...], ...},
    NOT flat dicts. Read the real fields so the reranker can actually compare them.
    """
    biz = bundle.get("business") or {}
    name = biz.get("business_name") or ""
    category = biz.get("category_name") or ""
    sub_category = biz.get("sub_category_name") or ""
    description = biz.get("message") or ""

    services = ""
    for highlight in (bundle.get("highlights") or []):
        services += (highlight.get("products_or_services") or "") + " "
        
    reviews = []
    for review in (bundle.get("reviews") or [])[:2]:
        text = review.get("review") or ""
        if text:
            reviews.append(text)
            
    faqs = []
    for faq in (bundle.get("faqs") or [])[:2]:
        q = faq.get("question") or ""
        a = faq.get("answer") or ""
        if q and a:
            faqs.append(f"Q: {q} A: {a}")

    parts = [
        f"Name: {name}",
        f"Category: {category}" + (f" / {sub_category}" if sub_category else ""),
        f"Description: {description}",
        f"Services: {services.strip()}",
        f"Reviews: {' | '.join(reviews)}",
        f"FAQs: {' | '.join(faqs)}"
    ]
    return " | ".join(p for p in parts if not p.endswith(": ") and not p.endswith(":"))


def rerank_with_scores(
    query: str,
    businesses: List[Dict[str, Any]],
    top_n: int,
) -> Tuple[List[Dict[str, Any]], List[Optional[float]]]:
    """
    Rerank business bundles by semantic relevance to the query.

    Returns (reranked_bundles, scores) where scores are the NVIDIA rerank logits
    aligned with reranked_bundles. A score is None when reranking was not actually
    performed (no API key, API error, or empty input) — callers must NOT treat a
    None score as a relevance signal.
    """
    nvidia_api_key = get_settings().nvidia_api_key
    if not nvidia_api_key or not businesses:
        sliced = businesses[:top_n]
        return sliced, [None] * len(sliced)

    # Create passages from business bundles (read nested fields, not top-level)
    passages = [{"text": _bundle_to_passage(b)} for b in businesses]

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "query": {"text": query},
        "passages": passages,
        "truncate": "END",
    }

    try:
        resp = requests.post(NVIDIA_RANKING_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Rankings: [{"index": 1, "logit": 1.43}, {"index": 0, "logit": -2.6}]
        # API returns them sorted by logit descending.
        rankings = data.get("rankings", [])

        reranked: List[Dict[str, Any]] = []
        scores: List[Optional[float]] = []
        for rank in rankings:
            idx = rank.get("index")
            if idx is None or idx >= len(businesses):
                continue
            reranked.append(businesses[idx])
            scores.append(float(rank.get("logit", 0.0)))
            if len(reranked) >= top_n:
                break

        # Safety: if parsing produced nothing, keep original order (no scores).
        if not reranked:
            sliced = businesses[:top_n]
            return sliced, [None] * len(sliced)
        return reranked, scores

    except Exception as exc:
        print(f"Reranker failed: {exc}")
        sliced = businesses[:top_n]
        return sliced, [None] * len(sliced)


def rerank_businesses(query: str, businesses: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper that returns only the reranked bundles."""
    reranked, _ = rerank_with_scores(query, businesses, top_n)
    return reranked
