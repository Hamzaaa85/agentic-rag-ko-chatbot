"""Shared LangGraph state (see docs/project__overview.md)."""

from typing import Any, TypedDict


class BusinessChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    history: list[dict[str, Any]]
    memory: dict[str, Any]

    plan: dict[str, Any] | None

    postgres_results: list[dict[str, Any]]
    pinecone_results: list[dict[str, Any]]
    business_ids: list[int]
    businesses: list[dict[str, Any]]

    answer: str
    errors: list[str]
