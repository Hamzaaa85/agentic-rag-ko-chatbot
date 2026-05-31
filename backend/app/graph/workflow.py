"""Compile LangGraph app."""

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from backend.app.graph.nodes import (
    fetch_business_details,
    generate_answer,
    load_memory,
    merge_results,
    plan_query,
    rerank_results,
    run_tools,
    save_memory,
)
from backend.app.graph.state import BusinessChatState
from backend.app.schemas.planner import SearchPlan


def route_after_plan(state: BusinessChatState) -> Literal["direct", "follow_up", "search"]:
    """Skip retrieval tools for direct replies and memory-only follow-ups."""
    plan = SearchPlan.model_validate(state.get("plan") or {})
    if plan.action == "chat":
        return "direct"
    if plan.action == "follow_up":
        return "follow_up"
    return "search"


@lru_cache
def build_graph():
    """Build and compile the business chat workflow."""
    graph = StateGraph(BusinessChatState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("plan_query", plan_query)
    graph.add_node("run_tools", run_tools)
    graph.add_node("merge_results", merge_results)
    graph.add_node("fetch_business_details", fetch_business_details)
    graph.add_node("rerank_results", rerank_results)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "plan_query")
    graph.add_conditional_edges(
        "plan_query",
        route_after_plan,
        {
            "direct": "generate_answer",
            "follow_up": "merge_results",
            "search": "run_tools",
        },
    )
    graph.add_edge("run_tools", "merge_results")
    graph.add_edge("merge_results", "fetch_business_details")
    graph.add_edge("fetch_business_details", "rerank_results")
    graph.add_edge("rerank_results", "generate_answer")
    graph.add_edge("generate_answer", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


def run_graph(session_id: str, message: str) -> BusinessChatState:
    """Invoke the compiled graph for one chat turn."""
    graph = build_graph()
    return graph.invoke(
        {
            "session_id": session_id,
            "user_message": message,
            "errors": [],
        }
    )

def stream_graph(session_id: str, message: str):
    """Stream the compiled graph for one chat turn, yielding messages and values."""
    graph = build_graph()
    return graph.stream(
        {
            "session_id": session_id,
            "user_message": message,
            "errors": [],
        },
        stream_mode=["messages", "values"]
    )
