import sys, os
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from backend.app.tools.pinecone_search import search_pinecone
import requests

api_key = os.getenv("NVIDIA_API_KEY")

queries = [
    ("cheap baby products", "Karachi"), # NikkaNikki should be high
    ("gym Nazimabad", "Karachi"),       # Elevate should be medium/high
    ("multani halwa", "Karachi"),       # All should be very low
    ("car mechanic", "Lahore"),         # If exists, high. Else low.
    ("wedding photographer", None)
]

print("=== RERANKER SCORE ANALYSIS ===")
for q, city in queries:
    print(f"\nQuery: '{q}' | City: {city}")
    hits = search_pinecone(q, top_k=5, city=city)
    if not hits:
        print("  No Pinecone hits.")
        continue
    
    passages = []
    seen = set()
    valid_hits = []
    for h in hits:
        if h.business_id not in seen:
            seen.add(h.business_id)
            passages.append({"text": f"Name: {h.business_name} | City: {h.city}"})
            valid_hits.append(h)
            
    if not passages: continue
    
    resp = requests.post(
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"},
        json={"model": "nvidia/rerank-qa-mistral-4b", "query": {"text": q}, "passages": passages, "truncate": "END"}
    )
    if resp.status_code == 200:
        data = resp.json().get("rankings", [])
        for r in data:
            print(f"  Score: {r['logit']:6.2f} | {passages[r['index']]['text']}")
    else:
        print("  API Error:", resp.status_code)
