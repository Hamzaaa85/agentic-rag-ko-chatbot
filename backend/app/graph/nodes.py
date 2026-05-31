"""Graph nodes: load_memory, plan_query, run_tools, merge, fetch_details, answer, save_memory."""

from __future__ import annotations

from typing import Any
import asyncio

from backend.app.graph.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_planner_user_prompt,
    get_planner_system_prompt,
)
from backend.app.config import get_settings
from backend.app.graph.state import BusinessChatState
from langchain_core.runnables import RunnableConfig

from backend.app.schemas.planner import SearchPlan
from backend.app.services.llm import get_chat_model
from backend.app.services.nvidia_rerank import rerank_with_scores
from backend.app.services.session_memory import get_session_memory, save_session_memory
from backend.app.tools import get_businesses_by_ids, search_businesses, search_pinecone


def _with_error(state: BusinessChatState, message: str) -> dict[str, Any]:
    return {"errors": [*state.get("errors", []), message]}


def _plan_from_state(state: BusinessChatState) -> SearchPlan:
    raw_plan = state.get("plan") or {}
    if isinstance(raw_plan, SearchPlan):
        return raw_plan
    return SearchPlan.model_validate(raw_plan)


def _cap_limit(value: Any, default: int = 5) -> int:
    try:
        return max(1, min(int(value), 10))
    except (TypeError, ValueError):
        return default


def load_memory(state: BusinessChatState) -> dict[str, Any]:
    """Load RAM-backed session context for the current turn."""
    session_id = state["session_id"]
    memory = get_session_memory(session_id)
    return {
        "memory": memory,
        "history": memory.get("history", []),
        "errors": state.get("errors", []),
    }


def plan_query(state: BusinessChatState) -> dict[str, Any]:
    """Ask the LLM for a strict structured plan."""
    memory = state.get("memory", {})
    model = get_chat_model().with_structured_output(SearchPlan, method="function_calling")
    prompt = build_planner_user_prompt(
        user_message=state["user_message"],
        history=state.get("history", []),
        summary=memory.get("summary", "No summary available."),
        last_business_names=memory.get("last_business_names", []),
        last_business_ids=memory.get("last_business_ids", [])
    )

    try:
        plan = model.invoke(
            [
                {"role": "system", "content": get_planner_system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        return {
            **_with_error(state, f"Planner failed: {exc}"),
            "plan": SearchPlan(needs_postgres=True).model_dump(),
        }

    return {"plan": plan.model_dump()}


import concurrent.futures

def run_tools(state: BusinessChatState) -> dict[str, Any]:
    """Run only the retrieval tools requested by the planner concurrently."""
    plan = _plan_from_state(state)
    if plan.action in {"direct_reply", "follow_up"}:
        return {"postgres_results": [], "pinecone_results": []}

    limit = _cap_limit(plan.limit)
    filters = dict(plan.filters or {})
    postgres_results: list[dict[str, Any]] = []
    pinecone_results: list[dict[str, Any]] = []
    errors = list(state.get("errors", []))

    def fetch_postgres():
        nonlocal postgres_results
        try:
            rows = search_businesses(filters=filters, limit=limit)
            postgres_results = [row.model_dump() for row in rows]
        except Exception as exc:
            errors.append(f"Postgres search failed: {exc}")

    def fetch_pinecone():
        nonlocal pinecone_results
        query = plan.semantic_query if plan.semantic_query else state["user_message"]
        raw_city = filters.get("city")
        pinecone_city = raw_city.strip().title() if raw_city and isinstance(raw_city, str) else None
        try:
            matches = search_pinecone(
                query,
                top_k=limit * 3,
                city=pinecone_city,
                category_id=filters.get("category_id"),
                sub_category_id=filters.get("sub_category_id"),
            )
            pinecone_results = [match.model_dump() for match in matches]
        except Exception as exc:
            errors.append(f"Pinecone search failed: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_postgres = executor.submit(fetch_postgres) if plan.needs_postgres else None
        f_pinecone = executor.submit(fetch_pinecone) if plan.needs_pinecone else None
        
        if f_postgres:
            f_postgres.result()
        if f_pinecone:
            f_pinecone.result()

    return {
        "postgres_results": postgres_results,
        "pinecone_results": pinecone_results,
        "errors": errors,
    }


def merge_results(state: BusinessChatState) -> dict[str, Any]:
    """
    Merge business IDs from Postgres and Pinecone.

    Priority order:
      1. IDs found by BOTH sources (highest confidence)
      2. Pinecone-only IDs when semantic search was used (relevance-ranked)
      3. Postgres-only IDs (structural matches, no relevance ranking)

    When only Postgres was used (needs_pinecone=false), Postgres order is kept.
    """
    plan = _plan_from_state(state)
    limit = _cap_limit(plan.limit)

    if plan.action == "chat":
        return {"business_ids": []}

    if plan.action == "follow_up":
        ids = state.get("memory", {}).get("last_business_ids", [])
        if not ids:
            return {
                "business_ids": [],
                "answer": "I don't have a previous business list for this session. Please search for businesses first.",
            }

        if plan.follow_up_business_ids:
            # Only keep the ones that were actually in the last search to prevent hallucinations
            selected = [i for i in plan.follow_up_business_ids if i in ids]
            if selected:
                return {"business_ids": selected}
                
        # Fall back to first business if no valid IDs found
        return {"business_ids": [int(ids[0])]}

    postgres_ids = [int(row["id"]) for row in state.get("postgres_results", []) if row.get("id")]
    pinecone_ids = [
        int(hit["business_id"])
        for hit in state.get("pinecone_results", [])
        if hit.get("business_id")
    ]

    postgres_set = set(postgres_ids)
    pinecone_set = set(pinecone_ids)
    ordered: list[int] = []

    # A "semantic" query expresses intent/quality/product (e.g. "multani halwa",
    # "cheap baby products"). For these, relevance must come from Pinecone/rerank —
    # Postgres city-only rows are NOT relevant and would hijack the answer.
    is_semantic = bool(plan.semantic_query)

    # Step 1: IDs found by BOTH sources — highest confidence
    for business_id in pinecone_ids:
        if business_id in postgres_set and business_id not in ordered:
            ordered.append(business_id)

    if plan.needs_pinecone:
        # Step 2: Pinecone-only IDs — semantically relevant, ranked by score
        for business_id in pinecone_ids:
            if business_id not in ordered:
                ordered.append(business_id)

        # Step 3: Postgres-only IDs are appended ONLY for non-semantic queries.
        # For semantic queries we deliberately drop them to avoid filler pollution
        # (e.g. random Karachi food shops showing up for "multani halwa").
        if not is_semantic:
            for business_id in postgres_ids:
                if business_id not in ordered:
                    ordered.append(business_id)
    else:
        # When only Postgres was used, keep Postgres order (no semantic data)
        for business_id in postgres_ids:
            if business_id not in ordered:
                ordered.append(business_id)

    # Over-fetch (more than limit) so the reranker has candidates to choose from.
    return {"business_ids": ordered[:limit * 3]}


def fetch_business_details(state: BusinessChatState) -> dict[str, Any]:
    """Fetch source-of-truth bundles from Postgres for final answering."""
    business_ids = state.get("business_ids", [])
    if not business_ids:
        return {"businesses": []}

    try:
        return {"businesses": get_businesses_by_ids(business_ids)}
    except Exception as exc:
        return {**_with_error(state, f"Business detail fetch failed: {exc}"), "businesses": []}


def rerank_results(state: BusinessChatState) -> dict[str, Any]:
    """Rerank fetched businesses and gate on relevance (abstain if nothing fits)."""
    plan = _plan_from_state(state)
    businesses = state.get("businesses", [])

    if not businesses or plan.action in ["direct_reply", "follow_up"]:
        return {}

    query = plan.semantic_query if plan.semantic_query else state["user_message"]
    limit = _cap_limit(plan.limit)

    reranked, scores = rerank_with_scores(query, businesses, top_n=limit)

    # Confidence gate (abstention) — OFF unless explicitly enabled in config.
    # When enabled: only for a semantic query where the reranker actually scored
    # results, and only if the best score is below the threshold. Disabled by
    # default so good results are never wrongly rejected before tuning with eval.
    settings = get_settings()
    if settings.rerank_abstain_enabled:
        real_scores = [s for s in scores if s is not None]
        if plan.semantic_query and real_scores:
            if max(real_scores) < settings.rerank_relevance_threshold:
                return {"businesses": [], "business_ids": [], "no_match": True}

    # Items are bundles: the id lives under bundle["business"]["id"], not top-level.
    new_ids: list[int] = []
    for bundle in reranked:
        biz = bundle.get("business") or {}
        if biz.get("id") is not None:
            new_ids.append(int(biz["id"]))
    return {
        "businesses": reranked,
        "business_ids": new_ids,
        "no_match": False,
    }


def _clarify_no_match(user_message: str) -> str:
    """Natural, language-matched 'not found / clarify' reply when nothing is relevant."""
    system = (
        "You are a friendly business-listings assistant. The user searched for "
        "something we have NO relevant match for. Reply in 1-2 short, warm sentences. "
        "Tell them you could not find a matching business for their request, and ask a "
        "brief clarifying question or suggest adding a city or trying a related term. "
        "Do NOT list or invent any business. "
        "Language: reply in English by default, but in Roman Urdu if the user wrote in "
        "Roman Urdu or Urdu."
    )
    try:
        response = get_chat_model().invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]
        )
        text = str(response.content).strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "I couldn't find a business matching that request. Could you add a city or "
        "try a related term? / Is request ke liye koi business nahi mila — city ya "
        "thori detail add karke dobara try karein."
    )


def generate_answer(state: BusinessChatState, config: RunnableConfig) -> dict[str, Any]:
    """Generate final answer from Postgres data only."""
    plan = _plan_from_state(state)
    if plan.action == "direct_reply":
        return {"answer": plan.answer or "You're welcome!"}

    businesses = state.get("businesses", [])

    # Abstention: retrieval/rerank found nothing relevant enough to show.
    if state.get("no_match") and not businesses and plan.action != "chat":
        return {"answer": _clarify_no_match(state["user_message"])}

    if state.get("answer") and not businesses and plan.action != "chat":
        return {"answer": state["answer"]}

    if not businesses and plan.action != "chat":
        return {"answer": _clarify_no_match(state["user_message"])}

    # Detail mode: follow_up means user asked about a specific business (contact/more info).
    # Chat mode is naturally conversational without detail formatting.
    detail_mode = plan.action == "follow_up" or (plan.action != "chat" and len(businesses) == 1)

    prompt = build_answer_user_prompt(
        user_message=state["user_message"],
        history=state.get("history", []),
        businesses=businesses,
        plan=plan.model_dump(),
        detail_mode=detail_mode,
    )

    try:
        response = get_chat_model().invoke(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            config=config
        )
    except Exception as exc:
        return {
            **_with_error(state, f"Answer generation failed: {exc}"),
            "answer": "Business data mil gaya, lekin final answer generate nahi ho saka. Please dobara try karein.",
        }

    return {"answer": str(response.content)}


def save_memory(state: BusinessChatState) -> dict[str, Any]:
    """Persist compact session memory for follow-up turns."""
    plan = _plan_from_state(state)
    memory = dict(state.get("memory", {}))
    history = list(state.get("history", []))
    history.extend(
        [
            {"role": "user", "content": state["user_message"]},
            {"role": "assistant", "content": state.get("answer", "")},
        ]
    )

    memory["history"] = history
    if plan.action == "business_search" and state.get("business_ids"):
        new_ids = list(state["business_ids"])
        new_names = [b.get("business", {}).get("business_name", "Unknown") for b in state.get("businesses", [])]
        
        combined_ids = []
        combined_names = []
        
        # Add new ones first
        for bid, bname in zip(new_ids, new_names):
            if bid not in combined_ids:
                combined_ids.append(bid)
                combined_names.append(bname)
                
        # Add existing ones from memory
        for bid, bname in zip(memory.get("last_business_ids", []), memory.get("last_business_names", [])):
            if bid not in combined_ids:
                combined_ids.append(bid)
                combined_names.append(bname)
                
        # Keep up to top 15 recent businesses in memory context
        memory["last_business_ids"] = combined_ids[:15]
        memory["last_business_names"] = combined_names[:15]
        memory["last_filters"] = dict(plan.filters or {})
    memory["last_plan"] = plan.model_dump()
    
    # --- MEMORY SUMMARIZATION ANTI-BLOAT ---
    # If we have more than 3 user-assistant pairs (6 messages), we summarize the oldest
    # to prevent context window bloat and LLM confusion.
    if len(memory["history"]) > 6:
        from backend.app.graph.prompts import SUMMARY_SYSTEM_PROMPT
        history_to_summarize = memory["history"][:-4]  # Keep last 2 pairs intact
        
        # Build prompt for summarizer
        lines = []
        if memory.get("summary"):
            lines.append(f"Previous Summary: {memory['summary']}")
        lines.append("Recent Messages:")
        for msg in history_to_summarize:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content')}")
            
        try:
            summary_response = get_chat_model().invoke([
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)}
            ])
            memory["summary"] = str(summary_response.content)
            # Truncate history to only the last 2 pairs (4 messages)
            memory["history"] = memory["history"][-4:]
        except Exception as exc:
            print(f"Summarization failed: {exc}") # Non-fatal

    saved = save_session_memory(state["session_id"], memory)
    return {
        "memory": saved,
        "history": saved.get("history", []),
    }
