import sys, json
sys.path.append('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot')
from dotenv import load_dotenv
load_dotenv('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot/.env')
from backend.app.graph.nodes import run_tools, merge_results, fetch_business_details, rerank_results
from backend.app.schemas.planner import SearchPlan
from backend.app.services.nvidia_rerank import rerank_with_scores
from backend.app.config import get_settings
print('Threshold:', get_settings().rerank_relevance_threshold)

state = {
    'user_message': 'gym',
    'plan': SearchPlan(action='business_search', filters={'city': 'Islamabad', 'category_id': 6}, semantic_query='gym', needs_pinecone=True, needs_postgres=True).model_dump()
}
s1 = run_tools(state)
state.update(s1)
s2 = merge_results(state)
state.update(s2)
s3 = fetch_business_details(state)
businesses = s3['businesses']

reranked, scores = rerank_with_scores('gym', businesses, top_n=5)
for b, s in zip(reranked, scores):
    print(f"{b['business']['business_name']} -> {s}")
