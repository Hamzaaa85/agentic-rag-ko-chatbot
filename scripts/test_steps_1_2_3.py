"""
Smoke test Steps 1–3 (from project root):

  python scripts/test_steps_1_2_3.py
  python scripts/test_steps_1_2_3.py --business-id 77
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas.search import BusinessSearchFilters
from backend.app.tools import (
    get_business_by_id,
    search_business_ids,
    search_pinecone,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--business-id", type=int, default=77)
    args = parser.parse_args()

    print("=== Step 1: get_business_by_id ===")
    bundle = get_business_by_id(args.business_id)
    if not bundle:
        print(f"No business for id={args.business_id}")
    else:
        print(f"OK: {bundle['business'].get('business_name')} (id={args.business_id})")

    print("\n=== Step 2: postgres search ===")
    ids = search_business_ids(
        filters=BusinessSearchFilters(city="Karachi", has_website=True),
        limit=5,
    )
    print(f"IDs: {ids}")

    print("\n=== Step 3: pinecone search ===")
    hits = search_pinecone("dairy milkshake Karachi", top_k=5, city="Karachi")
    print(json.dumps([h.model_dump() for h in hits], indent=2, default=str))


if __name__ == "__main__":
    main()
