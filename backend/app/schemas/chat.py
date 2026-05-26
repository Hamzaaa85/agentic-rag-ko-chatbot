"""Request/response models for /api/chat."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated conversation id")
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    business_ids: list[int] = []
