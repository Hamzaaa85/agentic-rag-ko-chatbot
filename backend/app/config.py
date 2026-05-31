"""Central settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    # openai_chat_model: str = "gpt-4o-mini"
    openai_chat_model: str = "gpt-5.4-mini-2026-03-17"
    # Cheaper model for the planner (routing/classification only — no prose).
    # gpt-4o-mini is fast, cheap, and excellent at structured output.
    planner_chat_model: str = "gpt-4o-mini"
    database_url: str = ""

    # Postgres pool (use with Neon -pooler URL for best results)
    db_pool_min: int = 2
    db_pool_max: int = 20

    pinecone_api_key: str = ""
    pinecone_index_name: str = "new-business-index"
    pinecone_host: str | None = None
    pinecone_namespace: str | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 1024

    # Reranking (NVIDIA NIM). When key is empty, reranking is skipped gracefully.
    # NVIDIA Reranker (Cross-Encoder)
    nvidia_api_key: str = ""
    # Abstention: filter out results whose reranker logit is significantly worse than the top match.
    # We use a relative dropoff margin rather than an absolute threshold because uncalibrated logits
    # fluctuate wildly depending on the query. A margin of 4.0 aggressively filters noise.
    rerank_abstain_enabled: bool = True
    rerank_dropoff_margin: float = 4.0

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
