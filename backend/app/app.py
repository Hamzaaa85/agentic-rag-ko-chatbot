"""Streamlit smoke-test UI for the LangGraph business chat workflow.

Run from the project root:
  streamlit run backend/app/app.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.app.graph.workflow import run_graph
from backend.app.services.session_memory import clear_session_memory, get_session_memory


def _json_pretty(value: Any) -> str:
    """Render debug payloads without failing on dates or custom objects."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _reset_chat() -> None:
    clear_session_memory(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.last_result = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.text_input(
            "Session ID",
            key="session_id",
            help="Same ID rakhein to follow-up questions memory use karengi.",
        )

        if st.button("Clear this session", use_container_width=True):
            _reset_chat()
            st.rerun()

        st.divider()
        st.subheader("Try prompts")
        st.caption("Pehle search karein, phir follow-up se memory check karein.")
        examples = [
            "Karachi me website wali businesses dikhao",
            "pehlay walay ka number do",
            "second wala website batao",
            "Lahore me best restaurant businesses",
        ]
        for example in examples:
            st.code(example, language=None)

        st.divider()
        st.subheader("Current Memory")
        memory = get_session_memory(st.session_state.session_id)
        st.json(memory)


def _render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_debug(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Debug")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Business IDs", len(result.get("business_ids", [])))
    with col2:
        st.metric("Errors", len(result.get("errors", [])))

    with st.expander("Plan", expanded=True):
        st.code(_json_pretty(result.get("plan", {})), language="json")

    with st.expander("Business IDs"):
        st.code(_json_pretty(result.get("business_ids", [])), language="json")

    with st.expander("Errors"):
        errors = result.get("errors", [])
        if errors:
            st.code(_json_pretty(errors), language="json")
        else:
            st.success("No graph errors returned.")

    with st.expander("Fetched Businesses"):
        st.code(_json_pretty(result.get("businesses", [])), language="json")


def main() -> None:
    st.set_page_config(page_title="Business Chat Tester", page_icon=":mag:", layout="wide")
    _init_session_state()

    st.title("Business Chat Tester")
    st.caption("Streamlit UI for testing the existing LangGraph flow with session memory.")

    _render_sidebar()
    _render_chat()

    prompt = st.chat_input("Business search ya follow-up message likhein...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Graph run ho raha hai..."):
                try:
                    result = run_graph(session_id=st.session_state.session_id, message=prompt)
                    answer = result.get("answer") or "No answer returned."
                except Exception as exc:
                    result = {
                        "session_id": st.session_state.session_id,
                        "answer": f"Error: {exc}",
                        "errors": [str(exc)],
                    }
                    answer = result["answer"]

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_result = result
        st.rerun()

    if st.session_state.last_result:
        _render_debug(st.session_state.last_result)


if __name__ == "__main__":
    main()
