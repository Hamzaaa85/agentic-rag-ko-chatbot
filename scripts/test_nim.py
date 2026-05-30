import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv(Path('.').resolve() / '.env')
api_key = os.getenv('NVIDIA_API_KEY')

url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

payload = {
    "model": "nvidia/rerank-qa-mistral-4b",
    "query": {"text": "affordable gym with trainer"},
    "passages": [
        {"text": "Expensive gym but has trainers."},
        {"text": "Cheap local gym with personal trainers."}
    ],
    "truncate": "END"
}

resp = requests.post(url, headers=headers, json=payload)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent=2))
else:
    print(resp.text)
