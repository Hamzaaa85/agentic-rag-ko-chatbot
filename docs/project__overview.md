# Agentic RAG Business Chatbot - Project Overview

## Purpose

This project is a scalable backend for an agentic RAG chatbot over business listings.

The chatbot should answer user queries about businesses using both structured database data and semantic search.

Core idea:

```text
Postgres = source of truth
Pinecone = semantic retrieval
LLM = planner and answer writer
LangGraph = workflow controller
```

## Current Data Scope

The main source table is:

```text
business_listings
```

The business listing already has linked/accessible related data:

```text
reviews
faqs
ctas
business_highlights
business_package_contents
categories
sub_categories
seo_data
```

`pinecone_dump_log` is not part of the chatbot answer flow. It can be ignored for search and response generation.

## Problem With Simple RAG

Simple RAG sends most user questions directly to Pinecone and tries to answer from retrieved chunks.

That is not enough for this use case because many queries need exact database filtering.

Examples:

```text
Lahore me dentists dikhao
website wali businesses dikhao
jin businesses ka Instagram hai
best SEO wali listing kaun si hai
pehlay walay ka number do
thank you
```

These should not all use the same retrieval path.

## Target Backend Stack

Recommended backend stack:

```text
Python
FastAPI
LangGraph
Neon Postgres
Pinecone
LLM provider
LangSmith
Sentry
```

Frontend can be:

```text
Next.js chat UI
session_id per conversation
optional streaming responses
```

## High-Level Flow

```text
User message
  ↓
FastAPI endpoint
  ↓
Load session memory
  ↓
LangGraph workflow
  ↓
LLM creates structured plan
  ↓
Run Postgres and/or Pinecone tools
  ↓
Merge and rank business IDs
  ↓
Fetch final business data from Postgres
  ↓
LLM writes final answer
  ↓
Save session memory
  ↓
Return response
```

## LangGraph Responsibility

LangGraph should not make the system complicated. Its job is to keep the workflow controlled.

It should handle:

```text
planning
tool execution
branching
result merging
final response generation
session memory updates
observability per step
```

## Initial Graph Nodes

Use this graph for the first scalable version:

```text
load_memory
plan_query
run_tools
merge_results
fetch_business_details
generate_answer
save_memory
```

This keeps the system scalable without making v1 too complex.

## State Design

State should include the current message, session history, plan, tool results, and final answer.

```python
class BusinessChatState(TypedDict):
    session_id: str
    user_message: str
    history: list

    plan: dict | None

    postgres_results: list[dict]
    pinecone_results: list[dict]
    business_ids: list[int]
    businesses: list[dict]

    answer: str
    errors: list[str]
```

The state should stay clean. Do not add many fields until they are needed.

## Session Memory

For v1, session memory can be in RAM.

If the backend restarts, memory can reset. That is acceptable for the first version.

Memory should store:

```python
{
    "history": [],
    "last_business_ids": [],
    "last_filters": {},
    "last_plan": {}
}
```

This allows follow-up questions like:

```text
pehlay walay ka number do
second wala website rakhta hai?
aur Karachi me dikhao
```

Later, memory can move to Redis or Postgres.

## LLM Planner

The LLM should decide what the user wants and output structured JSON.

The LLM should not directly write raw SQL.

Example business search plan:

```json
{
  "action": "business_search",
  "needs_postgres": true,
  "needs_pinecone": true,
  "filters": {
    "city": "Lahore",
    "category": "dentist",
    "has_website": true
  },
  "semantic_query": "dentist with good online presence and SEO",
  "limit": 5
}
```

Example direct chat plan:

```json
{
  "action": "direct_reply",
  "needs_postgres": false,
  "needs_pinecone": false,
  "answer": "You're welcome!"
}
```

## Tool Design

Initial tools:

```text
search_postgres
search_pinecone
get_businesses_by_ids
```

### search_postgres

Used for exact structured filtering.

Useful filters:

```text
city
category
sub_category
has_website
package_status
instagram available
facebook available
business_model
```

The LLM provides filters. The backend builds safe SQL.

### search_pinecone

Used for semantic matching.

Examples:

```text
good SEO
affordable salon
professional dental clinic
best online presence
family friendly restaurant
```

Pinecone should return:

```json
{
  "business_id": 123,
  "chunk_type": "seo_profile",
  "score": 0.87,
  "text": "matched chunk text"
}
```

### get_businesses_by_ids

Used before final answer generation.

Final answer must come from fresh Postgres data, not only Pinecone text.

## Pinecone Chunk Strategy

Each business should have multiple focused chunks.

Recommended chunk types:

```text
core_profile
seo_profile
reviews_profile
faqs_profile
highlights_profile
package_profile
contact_location
```

Every chunk must include:

```json
{
  "business_id": 123,
  "chunk_type": "core_profile",
  "business_name": "ABC Dental",
  "city": "Lahore",
  "category_id": 5,
  "sub_category_id": 12,
  "slug": "abc-dental"
}
```

Do not put every database column into metadata. Keep metadata for filtering and linking.

## Search Strategy

Use Postgres for exact filters.

Use Pinecone for semantic intent.

Use both for mixed queries.

Example:

```text
Lahore me dentists chahiye jin ki website bhi ho aur SEO acha ho
```

Flow:

```text
Postgres filters city/category/website
Pinecone searches SEO and relevance
Merge business IDs
Fetch full data from Postgres
Generate answer
```

## Result Merging

Initial merge logic:

```text
1. Prefer businesses found by both Postgres and Pinecone
2. Then include strong Postgres matches
3. Then include strong Pinecone matches
4. Cap results by requested limit
```

Later, a scoring function or reranker can be added.

## Answer Rules

The final answer should:

```text
use fetched Postgres business data
not invent phone numbers or websites
mention missing data clearly
respect user requested limit
ask clarification when query is vague
use session history for follow-ups
```

## Observability

Use LangSmith from the start.

Track:

```text
session_id
user_message
planner output
tools used
Postgres result count
Pinecone result count
final business_ids
latency
errors
```

Use Sentry for backend exceptions.

OpenTelemetry can be added later when dashboards and infra monitoring are needed.

## Scalability Path

V1:

```text
FastAPI
LangGraph
Neon Postgres
Pinecone
RAM session memory
LangSmith
```

V2:

```text
Redis or Postgres session memory
better ranking
streaming responses
evaluation dataset
admin/debug endpoints
```

V3:

```text
OpenTelemetry
Grafana/OpenObserve
advanced analytics
multi-tenant support if needed
```

## What To Avoid

Avoid:

```text
LLM writing raw SQL
answering only from Pinecone
hardcoded 3-rule intent router
putting full DB rows into Pinecone metadata
adding too many LangGraph state fields too early
```

## Final Architecture

```text
Next.js Chat UI
  ↓
FastAPI Backend
  ↓
LangGraph Workflow
  ↓
LLM Planner
  ↓
Postgres Tool + Pinecone Tool
  ↓
Merge Results
  ↓
Fetch Full Business Data
  ↓
LLM Final Answer
  ↓
LangSmith Trace
```

This is scalable because responsibilities are separated clearly.

Postgres owns truth. Pinecone owns semantic search. LLM owns planning and response writing. LangGraph owns workflow control.
