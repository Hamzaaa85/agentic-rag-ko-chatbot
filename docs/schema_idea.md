# Table: business_listings

1. **id** — `SERIAL PRIMARY KEY`  
2. **full_name** — `VARCHAR(100) NOT NULL`  
3. **business_name** — `VARCHAR(150) NOT NULL`  
4. **mobile_number** — `VARCHAR(20) NOT NULL`  
5. **whatsapp_number** — `VARCHAR(20)`  
6. **email** — `VARCHAR(150) NOT NULL`  
7. **has_website** — `BOOLEAN`  
8. **preferred_language** — `VARCHAR(50)`  
9. **business_address** — `TEXT`  
10. **created_at** — `TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP`  
11. **city** — `TEXT`  
12. **category_id** — `BIGINT`  
13. **package_status** — `package_status_enum DEFAULT 'Basic'`  
14. **sub_category_id** — `BIGINT`  
15. **message** — `TEXT`  
16. **facebook_social_link** — `TEXT`  
17. **instagram_social_link** — `TEXT`  
18. **website_url** — `TEXT`  
19. **business_model** — `TEXT`  
20. **status_enumu** — `TEXT`  
21. **updated_at** — `TIMESTAMP WITH TIME ZONE DEFAULT now()`  
22. **is_test_data** — `BOOLEAN DEFAULT false`  
23. **ai_status** — `ai_status_enums NOT NULL DEFAULT 'under_review'`  
24. **logo** — `TEXT`  
25. **slug** — `VARCHAR(255) UNIQUE`  
26. **source** — `source`  
27. **pinecone_dump_status** — `BOOLEAN DEFAULT false`

# Table: seo_data

1. **id** — `SERIAL PRIMARY KEY`  
2. **business_id** — `INTEGER`  
3. **meta_title** — `TEXT`  
4. **meta_description** — `TEXT`  
5. **focus_keywords** — `TEXT`  
6. **url_slug** — `TEXT`  
7. **h1** — `TEXT`  
8. **image_alt_text** — `TEXT`

# Table: pinecone_dump_log

1. **id** — `SERIAL PRIMARY KEY`  
2. **run_at** — `TIMESTAMP NOT NULL DEFAULT now()`  
3. **business_id** — `INTEGER`  
4. **status** — `VARCHAR(50) NOT NULL`  
5. **message** — `TEXT`  
6. **error_message** — `TEXT`  
7. **chunks_count** — `INTEGER`  
8. **vectors_upserted** — `INTEGER`  
9. **pinecone_index_name** — `VARCHAR(128)`