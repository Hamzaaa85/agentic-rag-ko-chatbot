import sys, os
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from backend.app.graph.workflow import build_graph

g = build_graph()

# Fake a long conversation
state = {
    "session_id": "test_summary_01",
    "user_message": "what about multani halwa?",
    "memory": {
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "i am looking for a dentist"},
            {"role": "assistant", "content": "here is a dentist"},
            {"role": "user", "content": "what about a gym in karachi?"},
            {"role": "assistant", "content": "here are gyms in karachi"},
            {"role": "user", "content": "cheap ones?"},
            {"role": "assistant", "content": "here are cheap gyms in karachi"}
        ],
        "summary": "User likes fitness."
    }
}

print("Running save_memory to trigger summarization...")
from backend.app.graph.nodes import save_memory

# Fake a plan to satisfy save_memory
state["plan"] = {"action": "business_search", "filters": {}, "semantic_query": ""}
new_state = save_memory(state)

print("\n--- NEW MEMORY STATE ---")
print("History length:", len(new_state["history"]))
print("Summary:", new_state["memory"].get("summary"))
