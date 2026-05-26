"""
Smoke test Phase 4 LangGraph workflow (from project root):

  python scripts/test_graph.py "Karachi me website wali businesses"
  python scripts/test_graph.py "thank you"
  python scripts/test_graph.py "Karachi me website wali businesses" "pehlay walay ka number do"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.graph.workflow import run_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="+", help="One or more messages to send to the graph")
    parser.add_argument("--session-id", default="cli-smoke-test")
    args = parser.parse_args()

    for message in args.messages:
        result = run_graph(session_id=args.session_id, message=message)
        print(f"> {message}")
        print(result.get("answer", ""))
        print()
        print(
            json.dumps(
                {
                    "session_id": result.get("session_id"),
                    "business_ids": result.get("business_ids", []),
                    "plan": result.get("plan"),
                    "errors": result.get("errors", []),
                },
                indent=2,
                default=str,
            )
        )
        print()


if __name__ == "__main__":
    main()
