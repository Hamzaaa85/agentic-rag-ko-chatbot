"""Structured JSON plan from the LLM router/planner."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchPlan(BaseModel):
    """Planner contract: structured intent only, never SQL."""

    action: Literal["business_search", "direct_reply", "follow_up"] = "business_search"
    needs_postgres: bool = False
    needs_pinecone: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    semantic_query: str | None = None
    limit: int = Field(default=5, ge=1, le=10)
    answer: str | None = None
    follow_up_index: int | None = Field(
        default=None,
        description="Zero-based index for a single follow-up business.",
    )
    follow_up_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Zero-based indices when the user asks about multiple specific businesses "
            "in one message. E.g. [3, 4] for the 4th and 5th businesses shown."
        ),
    )

    @field_validator("semantic_query", "answer", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _normalize_flags(self) -> "SearchPlan":
        if self.action == "direct_reply":
            self.needs_postgres = False
            self.needs_pinecone = False
            self.filters = {}
            self.semantic_query = None
            return self

        if self.action == "follow_up":
            self.needs_postgres = False
            self.needs_pinecone = False
            # limit reflects how many businesses will be fetched
            if self.follow_up_indices:
                self.limit = len(self.follow_up_indices)
            else:
                self.limit = 1
            return self

        if not self.needs_postgres and not self.needs_pinecone:
            self.needs_postgres = True
        return self
