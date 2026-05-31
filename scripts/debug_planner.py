import sys
sys.path.append('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot')
from dotenv import load_dotenv
load_dotenv('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot/.env')
from backend.app.services.llm import get_chat_model
from backend.app.graph.prompts import get_planner_system_prompt, build_planner_user_prompt

sys_prompt = get_planner_system_prompt()
user_prompt = build_planner_user_prompt("What's the best sweet shop in Pakistan", [], [], [])

model = get_chat_model()
res = model.invoke([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_prompt}
])
print(res.content)
