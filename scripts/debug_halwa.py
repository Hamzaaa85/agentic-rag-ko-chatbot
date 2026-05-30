"""Simulate the full pipeline for 'multani halwa' to see what's happening at each stage."""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from backend.app.tools.pinecone_search import search_pinecone

# What Pinecone returns for "multani halwa" WITHOUT city filter
print("=== PINECONE: 'multani halwa' (NO city filter) ===")
hits = search_pinecone("multani halwa", top_k=10)
seen = set()
for h in hits:
    if h.business_id not in seen:
        seen.add(h.business_id)
        print(f"  id={h.business_id:4d} | score={h.score:.3f} | {h.business_name} ({h.city})")

print()
print("=== PINECONE: 'multani halwa' (city=Karachi) ===")
hits2 = search_pinecone("multani halwa", top_k=10, city="Karachi")
seen2 = set()
for h in hits2:
    if h.business_id not in seen2:
        seen2.add(h.business_id)
        print(f"  id={h.business_id:4d} | score={h.score:.3f} | {h.business_name} ({h.city})")

# Check if there's a Multan business with halwa
print()
print("=== PINECONE: 'multani halwa' (city=Multan) ===")
hits3 = search_pinecone("multani halwa", top_k=10, city="Multan")
seen3 = set()
for h in hits3:
    if h.business_id not in seen3:
        seen3.add(h.business_id)
        print(f"  id={h.business_id:4d} | score={h.score:.3f} | {h.business_name} ({h.city})")

import requests, json
# Now test reranker scores for the Karachi results
api_key = os.getenv("NVIDIA_API_KEY")
if api_key and hits2:
    passages = []
    for h in hits2:
        if h.business_id not in set(x.business_id for x in hits2[:hits2.index(h)]):
            passages.append({"text": f"Name: {h.business_name} | City: {h.city}"})
    
    resp = requests.post(
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"model": "nvidia/rerank-qa-mistral-4b", "query": {"text": "multani halwa"}, "passages": passages[:5], "truncate": "END"}
    )
    if resp.status_code == 200:
        print()
        print("=== RERANKER SCORES ===")
        for r in resp.json().get("rankings", []):
            print(f"  index={r['index']} | logit={r['logit']:.3f} | {passages[r['index']]['text'][:60]}")
