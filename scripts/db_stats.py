import psycopg2, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.').resolve() / '.env')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Total businesses
cur.execute("SELECT COUNT(*) FROM business_listings WHERE ai_status = 'ai_done'")
total = cur.fetchone()[0]

# Total gyms
cur.execute("SELECT COUNT(*) FROM business_listings WHERE sub_category_id = 28")
total_gyms = cur.fetchone()[0]

# Gyms in Karachi
cur.execute("SELECT COUNT(*) FROM business_listings WHERE sub_category_id = 28 AND city ILIKE '%Karachi%'")
karachi_gyms = cur.fetchone()[0]

# All Karachi gyms
cur.execute("SELECT id, business_name, business_address, city FROM business_listings WHERE sub_category_id = 28 AND city ILIKE '%Karachi%'")
rows = cur.fetchall()

# Top cities
cur.execute("SELECT city, COUNT(*) as cnt FROM business_listings WHERE ai_status = 'ai_done' GROUP BY city ORDER BY cnt DESC LIMIT 10")
cities = cur.fetchall()

# Category distribution
cur.execute("""
SELECT c.name, COUNT(*) as cnt 
FROM business_listings b 
JOIN categories c ON b.category_id = c.id 
WHERE b.ai_status = 'ai_done'
GROUP BY c.name ORDER BY cnt DESC LIMIT 10
""")
cats = cur.fetchall()

# How many businesses have address with area/locality info?
cur.execute("SELECT COUNT(*) FROM business_listings WHERE business_address IS NOT NULL AND TRIM(business_address) <> '' AND ai_status = 'ai_done'")
with_address = cur.fetchone()[0]

print(f'=== DATABASE STATS ===')
print(f'Total businesses (ai_done): {total}')
print(f'Businesses with address: {with_address}')
print(f'Total gyms: {total_gyms}')
print(f'Gyms in Karachi: {karachi_gyms}')
print()
print(f'=== KARACHI GYMS ===')
for r in rows:
    print(f'  id={r[0]} | {r[1]} | {r[2]}')
print()
print(f'=== TOP CITIES ===')
for c in cities:
    print(f'  {c[0]}: {c[1]} businesses')
print()
print(f'=== TOP CATEGORIES ===')
for c in cats:
    print(f'  {c[0]}: {c[1]} businesses')
