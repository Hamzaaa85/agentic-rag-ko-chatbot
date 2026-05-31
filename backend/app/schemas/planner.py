"""Structured JSON plan from the LLM router/planner."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchPlan(BaseModel):
    """Planner contract: structured intent only, never SQL."""

    action: Literal["business_search", "chat", "follow_up", "direct_reply"] = "business_search"
    answer: str | None = Field(
        default=None,
        description="Instant reply string. Use ONLY when action is 'direct_reply' for simple greetings or thanks.",
    )
    needs_postgres: bool = False
    needs_pinecone: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    semantic_query: str | None = None
    limit: int = Field(default=5, ge=1, le=10)
    follow_up_business_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Exact Database IDs of businesses the user is asking to follow up on. "
            "E.g., [301] or [104, 45]. Do NOT guess IDs; use the exact ones provided in the conversation history."
        ),
    )

    @field_validator("semantic_query", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _normalize_flags(self) -> "SearchPlan":
        if self.action in {"chat", "direct_reply"}:
            self.needs_postgres = False
            self.needs_pinecone = False
            self.filters = {}
            self.semantic_query = None
            return self

        if self.action == "follow_up":
            self.needs_postgres = False
            self.needs_pinecone = False
            self.filters = {}
            self.semantic_query = None
            return self

        # Fallback safeguard: If LLM generated both false, compute them.
        # But if the LLM explicitly enabled one, we respect it.
        if not self.needs_postgres and not self.needs_pinecone:
            if self.filters:
                self.needs_postgres = True
            if self.semantic_query:
                self.needs_pinecone = True
            if not self.needs_postgres and not self.needs_pinecone:
                self.needs_pinecone = True

        return self
