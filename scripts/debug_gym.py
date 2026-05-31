import os, sys
sys.path.append('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot')
from dotenv import load_dotenv
load_dotenv('d:/Karobar-Online-Chatbots/agentic-rag-ko-chatbot/.env')
from backend.app.db.connection import get_db_cursor

with get_db_cursor() as cur:
    cur.execute("SELECT COUNT(id) FROM business_listings WHERE city ILIKE 'Lahore'")
    res = cur.fetchone()
    print('Lahore total:', res.get('count') if res else res)
    cur.execute("SELECT COUNT(id) FROM business_listings WHERE city ILIKE 'Lahore' AND sub_category_id = 28")
    res2 = cur.fetchone()
    print('Lahore gyms:', res2.get('count') if res2 else res2)
