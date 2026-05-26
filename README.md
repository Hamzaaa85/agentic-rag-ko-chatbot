# Agentic RAG — Business Listings Chatbot

Scalable backend for searching and answering questions over business listings (Neon Postgres + Pinecone + LangGraph).

## Project layout

```text
agentic-rag-trying/
├── backend/           # Chat API + LangGraph agent
├── scripts/           # One-off jobs (Pinecone dump)
├── docs/              # Architecture & schema notes
├── tests/             # Tests (add as you build)
├── .env               # Secrets (not committed)
├── .env.example       # Env template
└── requirements.txt
```

## Docs

- [Project overview](docs/project__overview.md) — architecture, state, tools, rollout
- [Tasks & pass criteria](docs/steps/tasks.md) — full checklist by phase
- [Schema notes](docs/schema_idea.md) — table reference

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then fill values
```

## Scripts

```bash
# Index businesses into Pinecone (from project root)
python scripts/pinecone_dump.py
```

## Backend (next)

```bash
uvicorn backend.app.main:app --reload
```

Implementation lives under `backend/app/` — see `docs/project__overview.md` for build order.
