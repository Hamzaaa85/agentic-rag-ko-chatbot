"""File-based session store for multi-worker safety without Redis."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, TypedDict

MAX_HISTORY_MESSAGES = 20
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


class SessionMemory(TypedDict, total=False):
    history: list[dict[str, str]]
    last_business_ids: list[int]
    last_business_names: list[str]
    last_filters: dict[str, Any]
    last_plan: dict[str, Any]
    summary: str


def _empty_memory() -> SessionMemory:
    return {
        "history": [],
        "last_business_ids": [],
        "last_business_names": [],
        "last_filters": {},
        "last_plan": {},
    }

def _get_session_path(session_id: str) -> str:
    # safe filename
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        safe_id = "default"
    return os.path.join(SESSIONS_DIR, f"{safe_id}.json")


def get_session_memory(session_id: str) -> SessionMemory:
    """Read session memory from disk."""
    path = _get_session_path(session_id)
    if not os.path.exists(path):
        return _empty_memory()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure it has empty arrays if missing
            mem = _empty_memory()
            mem.update(data)
            return mem
    except Exception:
        return _empty_memory()


def save_session_memory(session_id: str, memory: SessionMemory) -> SessionMemory:
    """Persist one session to disk and return a copy."""
    normalized = _empty_memory()
    normalized.update(memory)
    normalized["history"] = list(normalized.get("history", []))[-MAX_HISTORY_MESSAGES:]
    normalized["last_business_ids"] = [int(i) for i in normalized.get("last_business_ids", [])]
    normalized["last_business_names"] = list(normalized.get("last_business_names", []))
    normalized["last_filters"] = dict(normalized.get("last_filters", {}))
    normalized["last_plan"] = dict(normalized.get("last_plan", {}))
    
    path = _get_session_path(session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # fallback gracefully if file permissions fail
        
    return deepcopy(normalized)


def clear_session_memory(session_id: str) -> None:
    """Small testing helper for repeatable smoke tests."""
    path = _get_session_path(session_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
