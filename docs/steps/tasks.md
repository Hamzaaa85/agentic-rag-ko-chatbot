# Project Tasks & Pass Criteria

Agentic RAG chatbot over business listings (Neon Postgres + Pinecone + LangGraph).

**Principles**

```text
Postgres  = source of truth (final answers)
Pinecone  = semantic search (business_id discovery)
LLM       = planner + answer writer (no raw SQL)
LangGraph = controlled workflow
```

**Status legend:** `[x]` done · `[ ]` todo · `[~]` in progress

---

## Phase 0 — Foundation & docs

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P0-1 | Repo folder structure (`backend/`, `scripts/`, `docs/`, `tests/`) | [x] | `README.md` + `docs/STRUCTURE.md` match actual tree |
| P0-2 | `docs/project__overview.md` architecture doc | [x] | Describes flow, state, tools, chunking |
| P0-3 | `docs/schema_idea.md` + full SQL schema understood | [x] | `business_listings` + linked tables documented |
| P0-4 | `.env.example` + `requirements.txt` | [x] | New dev can `pip install -r requirements.txt` and copy `.env.example` |
| P0-5 | `docs/steps/tasks.md` (this file) | [x] | All phases listed with pass criteria |

---

## Phase 1 — Data indexing (Pinecone)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P1-1 | `scripts/pinecone_dump.py` — fetch full business bundle | [x] | One `business_id` loads listing + seo + highlights + package + faqs + reviews + ctas |
| P1-2 | Chunk strategy (profile chunks + metadata) | [x] | Chunks: `core_profile`, `seo_profile`, `highlights_profile`, `package_profile`, `faqs_profile`, `reviews_profile`, `cta_profile`, `contact_location` |
| P1-3 | Vector IDs readable | [x] | Format `business__{id}-{chunk-type}` (e.g. `business__77-core-profile`) |
| P1-4 | Eligibility query | [x] | Only `ai_status='ai_done'`, `pinecone_dump_status=false`, `EXISTS seo_data` |
| P1-5 | Upsert + mark `pinecone_dump_status=true` on success | [x] | After run, row flagged true; `pinecone_dump_log` has `success` row |
| P1-6 | Index auto-create (optional host) | [x] | Missing index creates serverless index; host resolved from API or `.env` |
| P1-7 | 404 namespace on delete (empty index) | [x] | First dump does not fail on delete-before-upsert |
| P1-8 | Bulk dump all eligible businesses | [ ] | `PINECONE_DUMP_LIMIT` removed; majority of `ai_done` rows indexed |
| P1-9 | Manual re-sync one business | [ ] | `dump_business_to_pinecone(business_id=…)` works after data change |

**Verify P1**

```powershell
python scripts/pinecone_dump.py
# Pinecone console: record count > 0, filter business_id=77 shows chunks
```

---

## Phase 2 — Database layer (Postgres tools)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P2-1 | `db/connection.py` — connection pool | [x] | `ThreadedConnectionPool`; `getconn`/`putconn`; no per-request `connect()`/`close()` |
| P2-2 | `db/queries.py` — SQL only | [x] | No OpenAI/Pinecone imports; parameterized queries only |
| P2-3 | `fetch_full_business_bundle` | [x] | Same shape as dump script for one `business_id` |
| P2-4 | `search_business_listings` + filters | [x] | Filters: city, category, website, social, package, `ai_status`, exclude test data |
| P2-5 | `tools/business_details.py` | [x] | `get_business_by_id`, `get_businesses_by_ids` |
| P2-6 | `tools/postgres_search.py` | [x] | `search_businesses`, `search_business_ids`; accepts dict filters from planner |
| P2-7 | `schemas/search.py` | [x] | `BusinessSearchFilters`, `BusinessListItem`, `PineconeMatch` |

**Verify P2**

```powershell
python scripts/test_steps_1_2_3.py --business-id 77
# Step 1: business name printed
# Step 2: list of IDs
```

**Optional `.env`**

```env
DB_POOL_MIN=2
DB_POOL_MAX=20
```

---

## Phase 3 — Vector search (Pinecone tool)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P3-1 | `services/embeddings.py` | [x] | Same model/dim as index (`text-embedding-3-large`, dim 1024) |
| P3-2 | `services/pinecone_client.py` | [x] | Cached index client; host normalized (no `https://`) |
| P3-3 | `tools/pinecone_search.py` | [x] | Returns `PineconeMatch` list with `business_id`, `chunk_type`, `score` |
| P3-4 | SDK v5 `QueryResponse` parsing | [x] | Uses `response.matches` + attribute access; not silent empty from `.get()` on objects |
| P3-5 | Optional metadata filters | [x] | `city`, `category_id`, `chunk_type` filter works in Pinecone query |
| P3-6 | `search_pinecone_business_ids` dedupe | [x] | Unique IDs in score order |

**Verify P3**

```powershell
python scripts/test_steps_1_2_3.py --business-id 77
# Step 3: JSON array with matches, scores > 0
```

---

## Phase 4 — LangGraph agent (core brain)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P4-1 | `graph/state.py` aligned with overview | [x] | Typed state: session, message, history, plan, tool results, `business_ids`, `businesses`, `answer`, `errors` |
| P4-2 | `services/llm.py` — chat model factory | [x] | OpenAI (or configured) model; used by planner + answer nodes |
| P4-3 | `graph/prompts.py` — planner + answer prompts | [x] | Planner outputs strict JSON (`SearchPlan`); no raw SQL in prompt |
| P4-4 | Node: `load_memory` | [x] | Loads RAM session by `session_id` |
| P4-5 | Node: `plan_query` | [x] | Sets `plan`: `action`, `needs_postgres`, `needs_pinecone`, `filters`, `semantic_query`, `limit` |
| P4-6 | Node: `run_tools` | [x] | Calls `postgres_search` / `pinecone_search` based on plan only |
| P4-7 | Node: `merge_results` | [x] | Merges IDs: prefer both sources, then SQL, then Pinecone; respects `limit` |
| P4-8 | Node: `fetch_business_details` | [x] | `get_businesses_by_ids` for merged IDs |
| P4-9 | Node: `generate_answer` | [x] | Answer uses only fetched Postgres data; no invented phone/URL |
| P4-10 | Node: `save_memory` | [x] | Updates `history`, `last_business_ids`, `last_filters` |
| P4-11 | `graph/workflow.py` — compile graph | [x] | Linear/conditional edges; smalltalk path skips tools |
| P4-12 | Follow-up resolution | [x] | "pehlay walay ka number" resolves via `last_business_ids[0]` |
| P4-13 | CLI test script | [x] | `python scripts/test_graph.py "Lahore me dentists"` returns coherent answer |

**Pass criteria (Phase 4 overall)**

| Test query | Expected behavior |
|------------|-------------------|
| `thank you` | Direct reply, no tools |
| `Karachi me website wali businesses` | Postgres filter + short list |
| `best SEO dairy Karachi` | Postgres + Pinecone hybrid |
| `pehlay walay ka number do` (after list) | Uses session memory + `get_business_by_id` |

---

## Phase 5 — Session memory

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P5-1 | `services/session_memory.py` — RAM store | [x] | Dict keyed by `session_id`; survives multiple turns same process |
| P5-2 | Fields: `history`, `last_business_ids`, `last_filters` | [x] | Follow-up tests pass |
| P5-3 | (Later) Persist via `chat_threads` table | [ ] | Optional v2; not required for v1 pass |

**Note:** v1 acceptable if restart clears memory (documented).

---

## Phase 6 — HTTP API (FastAPI)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P6-1 | `POST /api/chat` wired to graph | [ ] | Body: `session_id`, `message`; response: `answer`, `business_ids` |
| P6-2 | `schemas/chat.py` validation | [~] | Pydantic models exist; route uses them |
| P6-3 | `GET /health` | [x] | Returns `{"status":"ok"}` |
| P6-4 | Error handling | [ ] | 500 returns message; no secret leak in response |
| P6-5 | (Optional) API auth | [ ] | `API_PASSWORD` header check if required |

**Verify P6**

```powershell
uvicorn backend.app.main:app --reload
# POST http://127.0.0.1:8000/api/chat
```

---

## Phase 7 — Observability

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P7-1 | LangSmith env in `.env` | [~] | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` set |
| P7-2 | Trace graph runs | [ ] | Each chat shows planner output, tool calls, latency in LangSmith UI |
| P7-3 | (Later) Sentry for API errors | [ ] | Unhandled exceptions reported |
| P7-4 | (Later) OpenTelemetry / Grafana | [ ] | Optional production |

---

## Phase 8 — Testing & quality

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P8-1 | `scripts/test_steps_1_2_3.py` | [x] | All 3 steps pass for `business_id=77` |
| P8-2 | `tests/` unit tests for merge + filters | [ ] | pytest for SQL builder + ID merge |
| P8-3 | Evaluation query set (10–20 questions) | [ ] | Markdown list in `docs/` with expected behavior |
| P8-4 | Manual regression after chunk change | [ ] | Re-dump sample + re-run eval queries |

---

## Phase 9 — Frontend (later)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P9-1 | Next.js chat UI | [ ] | Generates `session_id` (UUID), calls `/api/chat` |
| P9-2 | Display answer + optional business cards | [ ] | Shows list from `business_ids` |
| P9-3 | (Optional) Streaming | [ ] | SSE from FastAPI |

---

## Phase 10 — Production hardening (later)

| ID | Task | Status | Pass criteria |
|----|------|--------|----------------|
| P10-1 | Full Pinecone index coverage | [ ] | All `ai_done` + seo businesses indexed |
| P10-2 | Rate limits / max `limit` cap | [ ] | Planner cannot request 1000 rows |
| P10-3 | Redis or `chat_threads` persistence | [ ] | Sessions survive deploy |
| P10-4 | Deploy FastAPI (Docker / Railway / etc.) | [ ] | HTTPS endpoint live |
| P10-5 | Secrets only in env, not git | [x] | `.gitignore` includes `.env` |

---

## Current priority order

```text
1. Phase 6 — /api/chat wire-up
2. Phase 7 — LangSmith traces on real chats
3. Phase 1 P1-8 — bulk Pinecone dump
4. Phase 8 — eval query set + pytest
5. Phase 9 — frontend
```

---

## Quick command reference

| Action | Command |
|--------|---------|
| Index businesses | `python scripts/pinecone_dump.py` |
| Test tools 1–3 | `python scripts/test_steps_1_2_3.py --business-id 77` |
| Run API | `uvicorn backend.app.main:app --reload` |
| Health check | `GET /health` |

---

## Definition of Done (whole project v1)

All must pass:

1. User can chat via API with `session_id` and get accurate answers from **Postgres data**.
2. Hybrid queries use **Postgres filters** + **Pinecone semantic** search.
3. Follow-up questions use **session memory** (`last_business_ids`).
4. No raw SQL from LLM; filters are structured JSON only.
5. LangSmith shows traces for debugging.
6. Majority of production businesses indexed in Pinecone (`ai_done` + seo).

---

## Related docs

- [Project overview](../project__overview.md)
- [Folder structure](../STRUCTURE.md)
- [Schema notes](../schema_idea.md)
