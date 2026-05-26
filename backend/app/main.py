"""
FastAPI entrypoint.

Run from project root:
  uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI

from backend.app.api.routes.chat import router as chat_router

app = FastAPI(
    title="Agentic RAG — Business Listings",
    version="0.1.0",
)

app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
