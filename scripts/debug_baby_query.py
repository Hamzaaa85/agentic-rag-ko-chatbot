"""Quick debug: check what Pinecone actually returns for 'cheap baby products in Karachi'."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.app.tools import search_pinecone, search_businesses

print("=" * 60)
print("PINECONE: 'cheap baby products' + city=karachi")
print("=" * 60)
hits = search_pinecone("cheap baby products", top_k=10, city="karachi")
if not hits:
    print(">>> ZERO PINECONE RESULTS! <<<")
    print("Trying WITHOUT city filter...")
    hits = search_pinecone("cheap baby products", top_k=10)

for h in hits:
    print(f"  id={h.business_id:4d} | score={h.score:.3f} | {h.chunk_type:20s} | {h.business_name} ({h.city})")

print()
print("=" * 60)
print("POSTGRES: city=Karachi (no category filter)")
print("=" * 60)
rows = search_businesses(filters={"city": "Karachi"}, limit=5)
for r in rows:
    print(f"  id={r.id:4d} | {r.business_name:30s} | {r.category_name} | {r.city}")

print()
pinecone_ids = [h.business_id for h in hits]
postgres_ids = [r.id for r in rows]
both = set(pinecone_ids) & set(postgres_ids)
print(f"Pinecone IDs: {pinecone_ids[:10]}")
print(f"Postgres IDs: {postgres_ids}")
print(f"Overlap: {both}")
print(f"Pinecone-only: {[x for x in pinecone_ids if x not in postgres_ids][:10]}")
