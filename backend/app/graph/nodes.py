"""Graph nodes: load_memory, plan_query, run_tools, merge, fetch_details, answer, save_memory."""

from __future__ import annotations

from typing import Any

from backend.app.graph.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_planner_user_prompt,
)
from backend.app.graph.state import BusinessChatState
from backend.app.schemas.planner import SearchPlan
from backend.app.services.llm import get_chat_model
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
        last_business_ids=memory.get("last_business_ids", []),
        last_filters=memory.get("last_filters", {}),
    )

    try:
        plan = model.invoke(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        return {
            **_with_error(state, f"Planner failed: {exc}"),
            "plan": SearchPlan(needs_postgres=True).model_dump(),
        }

    return {"plan": plan.model_dump()}


def run_tools(state: BusinessChatState) -> dict[str, Any]:
    """Run only the retrieval tools requested by the planner."""
    plan = _plan_from_state(state)
    if plan.action in {"direct_reply", "follow_up"}:
        return {"postgres_results": [], "pinecone_results": []}

    limit = _cap_limit(plan.limit)
    filters = dict(plan.filters or {})
    postgres_results: list[dict[str, Any]] = []
    pinecone_results: list[dict[str, Any]] = []
    errors = list(state.get("errors", []))

    if plan.needs_postgres:
        try:
            rows = search_businesses(filters=filters, limit=limit)
            postgres_results = [row.model_dump() for row in rows]
        except Exception as exc:
            errors.append(f"Postgres search failed: {exc}")

    if plan.needs_pinecone:
        query = (
            f"{plan.semantic_query} {state['user_message']}"
            if plan.semantic_query
            else state["user_message"]
        )
        # Normalize city to Title Case for Pinecone exact-match filter.
        # Postgres uses ILIKE (case-insensitive) so it does not need this.
        raw_city = filters.get("city")
        pinecone_city = raw_city.strip().title() if raw_city and isinstance(raw_city, str) else None
        try:
            matches = search_pinecone(
                query,
                top_k=limit * 3,
                city=pinecone_city,
                category_id=filters.get("category_id"),
            )
            pinecone_results = [match.model_dump() for match in matches]
        except Exception as exc:
            errors.append(f"Pinecone search failed: {exc}")

    return {
        "postgres_results": postgres_results,
        "pinecone_results": pinecone_results,
        "errors": errors,
    }


def merge_results(state: BusinessChatState) -> dict[str, Any]:
    """Merge IDs: both sources first, then Postgres, then Pinecone."""
    plan = _plan_from_state(state)
    limit = _cap_limit(plan.limit)

    if plan.action == "direct_reply":
        return {"business_ids": []}

    if plan.action == "follow_up":
        ids = state.get("memory", {}).get("last_business_ids", [])
        if not ids:
            return {
                "business_ids": [],
                "answer": "I don't have a previous business list for this session. Please search for businesses first.",
            }

        # Multiple businesses requested in one message (e.g. "KidKit details + Dost Bazaar number")
        if plan.follow_up_indices:
            selected: list[int] = []
            for idx in plan.follow_up_indices:
                if 0 <= idx < len(ids) and int(ids[idx]) not in selected:
                    selected.append(int(ids[idx]))
            # Fall back to first business if all indices were out of range
            return {"business_ids": selected if selected else [int(ids[0])]}

        # Single business follow-up
        index = plan.follow_up_index if plan.follow_up_index is not None else 0
        if index < 0 or index >= len(ids):
            index = 0
        return {"business_ids": [int(ids[index])]}

    postgres_ids = [int(row["id"]) for row in state.get("postgres_results", []) if row.get("id")]
    pinecone_ids = [
        int(hit["business_id"])
        for hit in state.get("pinecone_results", [])
        if hit.get("business_id")
    ]

    postgres_set = set(postgres_ids)
    pinecone_set = set(pinecone_ids)
    ordered: list[int] = []

    for business_id in postgres_ids:
        if business_id in pinecone_set and business_id not in ordered:
            ordered.append(business_id)

    for business_id in postgres_ids:
        if business_id not in ordered:
            ordered.append(business_id)

    for business_id in pinecone_ids:
        if business_id not in postgres_set and business_id not in ordered:
            ordered.append(business_id)

    return {"business_ids": ordered[:limit]}


def fetch_business_details(state: BusinessChatState) -> dict[str, Any]:
    """Fetch source-of-truth bundles from Postgres for final answering."""
    business_ids = state.get("business_ids", [])
    if not business_ids:
        return {"businesses": []}

    try:
        return {"businesses": get_businesses_by_ids(business_ids)}
    except Exception as exc:
        return {**_with_error(state, f"Business detail fetch failed: {exc}"), "businesses": []}


def generate_answer(state: BusinessChatState) -> dict[str, Any]:
    """Generate final answer from Postgres data only."""
    plan = _plan_from_state(state)
    if plan.action == "direct_reply":
        return {"answer": plan.answer or "You're welcome!"}

    businesses = state.get("businesses", [])

    if state.get("answer") and not businesses:
        return {"answer": state["answer"]}

    if not businesses:
        return {
            "answer": (
                "Koi matching business nahi mila. Aap city, category ya filters thore broad kar ke try karein."
            )
        }

    # Detail mode: follow_up means user asked about a specific business (contact/more info).
    # Also use detail mode when only one business is returned (nothing to list briefly).
    detail_mode = plan.action == "follow_up" or len(businesses) == 1

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
            ]
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
        memory["last_business_ids"] = list(state["business_ids"])
        memory["last_filters"] = dict(plan.filters or {})
    memory["last_plan"] = plan.model_dump()

    saved = save_session_memory(state["session_id"], memory)
    return {
        "memory": saved,
        "history": saved.get("history", []),
    }
