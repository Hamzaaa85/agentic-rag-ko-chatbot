"""POST /api/chat — wired in phase 2 when LangGraph is ready."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/chat")
def chat() -> dict[str, str]:
    return {
        "message": "Chat agent not implemented yet. Build graph in backend/app/graph/.",
    }
