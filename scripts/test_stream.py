import sys, os
from pathlib import Path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
from backend.app.graph.workflow import build_graph

g = build_graph()
inputs = {"session_id": "test_stream", "user_message": "hello", "errors": []}
for event_type, data in g.stream(inputs, stream_mode=["messages", "values"]):
    print("Event Type:", event_type)
    if event_type == "messages":
        msg, meta = data
        print("  Msg Meta:", meta)
        print("  Msg Content:", type(msg.content))
    elif event_type == "values":
        print("  Values keys:", data.keys())
