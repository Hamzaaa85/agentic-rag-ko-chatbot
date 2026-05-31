"""LLM client factory."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from backend.app.config import get_settings


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """Shared chat model for final answer generation (higher quality)."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        streaming=True,
    )


@lru_cache
def get_planner_model() -> ChatOpenAI:
    """Cheaper model for the planner (routing/classification only — no prose).

    Structured output with function_calling works great on gpt-4o-mini and
    it's significantly faster and cheaper than the answer model.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return ChatOpenAI(
        model=settings.planner_chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        streaming=False,  # planner doesn't need streaming
    )

