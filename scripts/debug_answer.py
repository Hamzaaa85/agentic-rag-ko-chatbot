import sys
sys.path.append('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot')
from dotenv import load_dotenv
load_dotenv('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot/.env')
from backend.app.services.llm import get_chat_model
from backend.app.graph.prompts import ANSWER_SYSTEM_PROMPT, _format_businesses_for_prompt
from backend.app.tools.business_details import get_businesses_by_ids

businesses = get_businesses_by_ids([348, 1335, 1316])
formatted = _format_businesses_for_prompt(businesses)

sys_prompt = ANSWER_SYSTEM_PROMPT
user_prompt = f"User message:\\nWhat is the best sweet shop in Pakistan?\\n\\nFetched business data:\\n{formatted}"

model = get_chat_model()
res = model.invoke([
    {'role': 'system', 'content': sys_prompt},
    {'role': 'user', 'content': user_prompt}
])
print(res.content)
