"""In-memory session store (v1). Upgrade to Redis or chat_threads table later."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

MAX_HISTORY_MESSAGES = 20


class SessionMemory(TypedDict, total=False):
    history: list[dict[str, str]]
    last_business_ids: list[int]
    last_filters: dict[str, Any]
    last_plan: dict[str, Any]


_sessions: dict[str, SessionMemory] = {}


def _empty_memory() -> SessionMemory:
    return {
        "history": [],
        "last_business_ids": [],
        "last_filters": {},
        "last_plan": {},
    }


def get_session_memory(session_id: str) -> SessionMemory:
    """Return a copy of the memory for one chat session."""
    if session_id not in _sessions:
        _sessions[session_id] = _empty_memory()
    return deepcopy(_sessions[session_id])


def save_session_memory(session_id: str, memory: SessionMemory) -> SessionMemory:
    """Persist one session in process memory and return a copy."""
    normalized = _empty_memory()
    normalized.update(memory)
    normalized["history"] = list(normalized.get("history", []))[-MAX_HISTORY_MESSAGES:]
    normalized["last_business_ids"] = [int(i) for i in normalized.get("last_business_ids", [])]
    normalized["last_filters"] = dict(normalized.get("last_filters", {}))
    normalized["last_plan"] = dict(normalized.get("last_plan", {}))
    _sessions[session_id] = normalized
    return deepcopy(normalized)


def clear_session_memory(session_id: str) -> None:
    """Small testing helper for repeatable smoke tests."""
    _sessions.pop(session_id, None)
