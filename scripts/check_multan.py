import os, psycopg2
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print("Multan Halwa:")
cur.execute("SELECT id, business_name, city FROM business_listings WHERE city='Multan' AND message ILIKE '%halwa%'")
print(cur.fetchall())

print("All Multan Food (Category 4):")
cur.execute("SELECT id, business_name, city FROM business_listings WHERE city='Multan' AND category_id=4")
print(cur.fetchall())

print("Top Multan Businesses by name:")
cur.execute("SELECT id, business_name, city FROM business_listings WHERE city='Multan' LIMIT 10")
print(cur.fetchall())
