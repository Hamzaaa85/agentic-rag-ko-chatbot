# Folder structure

```text
agentic-rag-trying/
├── backend/app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings from .env
│   ├── api/routes/chat.py      # HTTP endpoints
│   ├── graph/                  # LangGraph workflow
│   │   ├── state.py
│   │   ├── workflow.py
│   │   ├── nodes.py
│   │   └── prompts.py
│   ├── tools/                  # Postgres + Pinecone + detail fetch
│   ├── services/               # LLM, embeddings, session memory
│   ├── db/                     # Connection + SQL
│   └── schemas/                # Pydantic API + planner models
├── scripts/
│   └── pinecone_dump.py        # Indexing job (already working)
├── docs/
│   ├── project__overview.md
│   ├── schema_idea.md
│   ├── STRUCTURE.md
│   └── steps/tasks.md
└── tests/
```

## Build order

1. ~~`db/queries.py` + `tools/business_details.py`~~ done
2. ~~`tools/postgres_search.py` + `tools/pinecone_search.py`~~ done
3. `graph/` nodes + `workflow.py` — next
4. Wire `api/routes/chat.py`
5. LangSmith + tests

Test steps 1–3: `python scripts/test_steps_1_2_3.py --business-id 77`
