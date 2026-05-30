import psycopg2
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path('.').resolve() / '.env')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("SELECT id, business_name, city FROM business_listings WHERE business_name ILIKE '%Hamza Store%' OR business_name ILIKE '%BioAqua%'")
for row in cur.fetchall():
    print(row)
