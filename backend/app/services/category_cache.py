"""
Category cache — load once from DB, reuse in every planner call.

Why this file exists:
  The LLM planner needs to know the real category names and IDs from the
  database. Without this, it guesses category names like "dentist" while
  the DB has "Health & Wellness" — causing zero results.

  This module loads all active categories at first use, caches them in
  process memory, and provides a formatted prompt snippet the planner
  can reference.

Usage:
  from backend.app.services.category_cache import get_categories_for_prompt
  prompt_text = get_categories_for_prompt()   # ~750 tokens, cached
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.app.db.connection import get_db_cursor


# ── raw data ──────────────────────────────────────────────────────────

SQL_ALL_CATEGORIES = """
    SELECT id, name, parent_id, deals_with, helps_with
    FROM categories
    WHERE is_active = true
    ORDER BY parent_id NULLS FIRST, id
"""


@lru_cache
def load_categories() -> list[dict[str, Any]]:
    """Load all active categories once. Cached for process lifetime."""
    with get_db_cursor() as cur:
        cur.execute(SQL_ALL_CATEGORIES)
        rows = cur.fetchall()
    return [dict(row) for row in rows]


# ── formatted for planner prompt ──────────────────────────────────────

def _format_keywords(deals: list[str] | None, helps: list[str] | None) -> str:
    """Merge deals_with + helps_with into a compact keyword line."""
    items: list[str] = []
    if deals:
        items.extend(deals[:5])          # cap at 5 to save tokens
    if helps:
        items.extend(helps[:3])          # top 3 helps
    if not items:
        return ""
    return " | ".join(items)


@lru_cache
def get_categories_for_prompt() -> str:
    """
    Build a compact text block listing every category for the planner prompt.

    Output shape (~750 tokens):
      PARENT CATEGORIES:
      - id=4: Food & Beverage  [keywords: Snacks, Bakery, Beverages, ...]
      ...
      SUB-CATEGORIES (prefer these when a specific type matches):
      - id=28: Gyms  (parent: Health & Wellness)
      ...
    """
    categories = load_categories()

    parents = [c for c in categories if c["parent_id"] is None]
    subs = [c for c in categories if c["parent_id"] is not None]

    # Build parent name lookup for sub-category display
    parent_names: dict[int, str] = {c["id"]: c["name"] for c in parents}

    lines: list[str] = []

    lines.append("PARENT CATEGORIES:")
    for cat in parents:
        kw = _format_keywords(cat.get("deals_with"), cat.get("helps_with"))
        line = f"- id={cat['id']}: {cat['name']}"
        if kw:
            line += f"  [keywords: {kw}]"
        lines.append(line)

    lines.append("")
    lines.append("SUB-CATEGORIES (more specific — prefer these when a match exists):")
    for cat in subs:
        parent_name = parent_names.get(cat["parent_id"], "?")
        kw = _format_keywords(cat.get("deals_with"), cat.get("helps_with"))
        line = f"- id={cat['id']}: {cat['name']}  (parent: {parent_name})"
        if kw:
            line += f"  [keywords: {kw}]"
        lines.append(line)

    return "\n".join(lines)
