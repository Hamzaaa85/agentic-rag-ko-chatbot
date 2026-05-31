"""LLM client factory."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from backend.app.config import get_settings


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """Shared chat model for planner and final answer generation."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
        streaming=True,
    )
