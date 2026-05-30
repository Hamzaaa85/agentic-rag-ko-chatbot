import os, psycopg2
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM business_listings WHERE city ILIKE '%Sargodha%'")
print("Total Sargodha:", cur.fetchone()[0])

cur.execute("SELECT business_name, message FROM business_listings WHERE city ILIKE '%Sargodha%' AND category_id = 4")
rows = cur.fetchall()
print(f"Food places ({len(rows)}):")
for r in rows:
    print(r[0])
