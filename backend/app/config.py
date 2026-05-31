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
    # Abstention drops results if the highest reranker score for a semantic query
    # logit is below rerank_relevance_threshold. This is OFF by default because
    # the right threshold depends on the model's score distribution and must be
    # found with an eval set first — otherwise it wrongly rejects good results.
    rerank_abstain_enabled: bool = False
    rerank_relevance_threshold: float = -5.0

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
