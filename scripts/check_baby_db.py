import psycopg2
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path('.').resolve() / '.env')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("SELECT id, business_name FROM business_listings WHERE city ILIKE '%Karachi%' AND (business_name ILIKE '%baby%' OR business_name ILIKE '%kid%')")
rows = cur.fetchall()
print(f"Total baby/kid businesses in Karachi: {len(rows)}")
for r in rows:
    print(r)
