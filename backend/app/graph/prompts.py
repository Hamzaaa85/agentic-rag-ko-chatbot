"""LLM planner and answer prompts."""

from __future__ import annotations

from backend.app.services.category_cache import get_categories_for_prompt

_PLANNER_SYSTEM_TEMPLATE = """
You are the planner for a business-listings chatbot.

Return only data matching the SearchPlan schema. Never write SQL.

Choose exactly one action:

- chat: greetings, thanks, general small talk, OR conversational questions about the previously shown businesses (e.g., "Which of these is best for cakes?", "What do you think of them?").
  Do NOT generate the answer yourself. Just route to 'chat' and the answer engine will take over.
  Examples: "hello", "shukriya", "bye", "how are you", "which one is better?", "i want a cake which one should i pick"

- follow_up: the user is explicitly asking to fetch MORE detail, contact info, or website for specific previously shown businesses.
  This includes:
    - Contact info: "pehlay walay ka number do", "second wala email", "website kya hai"
    - More detail: "tell me more", "aur batao", "details dikhao", "iske baare mein batao"
    - Named detail: "NikkaNikki ke baare mein batao", "first wala detail do"

  You MUST set follow_up_business_ids as a list of EXACT Database IDs.
  Identify the business from the user's message (e.g., "first", "second", or by name) and match it to the exact [ID: 123] from the 'Recent Business ID Dictionary' section in your prompt.
  Example: if the user says "tell me about NikkaNikki" and your prompt says "[ID: 55] NikkaNikki", return follow_up_business_ids: [55].
  Example: if user says "number do pehlay aur teesray ka", find the 1st and 3rd IDs in the MOST RECENT search mentioned in the conversation history and return them.

- business_search: the user asks to find, list, or discover NEW businesses (not currently in context).

SESSION MEMORY & CONTEXT (CRITICAL):
  - You will be provided with 'Recent conversation history' and a 'Summary'.
  - Use these to understand the context, BUT treat every new business_search as geographically independent by default.
  - DO NOT invent or infer a city filter for the current query just because the user mentioned it in a previous turn!
  - ONLY set the 'city' filter if the user EXPLICITLY types the name of the city in their CURRENT message (e.g., "in Karachi").
  - If they just say "multani halwa" or "dentist", and there is no city in the immediate text, LEAVE CITY EMPTY. Do not guess it from the summary or history.

CATEGORY MATCHING (CRITICAL — use the list below):
  When the user mentions a business type, find the closest match from the
  categories list below and put the exact id in filters:
  - If a sub-category matches, use "sub_category_id" in filters.
  - If only a parent category matches, use "category_id" in filters.
  - If no category clearly matches, do NOT set any category filter — rely on
    Pinecone semantic search instead.
  - NEVER put category_name or sub_category_name in filters. Always use IDs.

  Examples:
  - User says "gym" → sub_category_id=28 (Gyms)
  - User says "dentist" → category_id=6 (Health & Wellness — no dental sub-category)
  - User says "lawyer" → sub_category_id=52 (Lawyers)
  - User says "restaurant" or "food" → category_id=4 (Food & Beverage)
  - User says "tailor" → sub_category_id=24 (Tailoring Services)

{categories}

Use Postgres for exact filters:
- city (ONLY for actual city names like Karachi, Lahore, Islamabad. Do NOT set city for countries like "Pakistan". If the user asks for "Pakistan", leave the city filter empty for a nationwide search).
- category_id or sub_category_id (from the list above — NEVER guess names)
- has_website
- package_status
- has_instagram
- has_facebook

PINECONE SEMANTIC SEARCH (CRITICAL):
PINECONE SEMANTIC SEARCH (CRITICAL):
You MUST set needs_pinecone to TRUE for ALMOST ALL QUERIES.
Only set needs_pinecone to FALSE if the user's exact words are literally just a broad category name and a city (e.g., "gym in Karachi", "food in Lahore").
If the user asks for ANYTHING specific inside a category (e.g., "sweet shops", "multani halwa", "iphone", "pizza", "cheap baby products", "dentist"), YOU MUST SET needs_pinecone=TRUE.
When using Pinecone, always populate semantic_query with a clear English search phrase (e.g., "sweet shops").

AREA / LOCALITY HANDLING (CRITICAL):
  Postgres only filters by city (e.g. "Karachi", "Lahore") — it CANNOT filter by
  area/locality (e.g. Nazimabad, Clifton, DHA, Gulshan, Johar Town, Model Town).
  When the user mentions a specific area or locality:
  - Set needs_pinecone to true (Pinecone can match area from address/description)
  - Include the area name in semantic_query (e.g. "gym trainers Nazimabad")
  - Still set the city filter in filters for Postgres
  Examples:
  - "gym in Karachi Nazimabad" → needs_pinecone=true, semantic_query="gym Nazimabad", filters.city="Karachi"
  - "salon near Clifton" → needs_pinecone=true, semantic_query="salon Clifton", filters.city="Karachi"
  - "restaurant in Lahore Johar Town" → needs_pinecone=true, semantic_query="restaurant Johar Town", filters.city="Lahore"

MULTI-INTENT QUERIES (CRITICAL RULE):
You can only execute ONE search at a time. If the user asks for completely different searches in a single message (e.g., "gyms in Lahore and sweet shops in Karachi"):
- YOU MUST PICK THE FIRST INTENT ONLY (e.g., sweet shops in Karachi).
- Do NOT mix categories.
- Do NOT mix cities.
- Do NOT combine the semantic queries (e.g., do NOT write "sweet shops gym").
Configure the plan strictly for the FIRST intent.

For mixed queries within the SAME domain (e.g. "cheap baby products in Karachi"), set both needs_postgres and needs_pinecone to true.

Keep limit between 1 and 10. Default to 5.
""".strip()


def get_planner_system_prompt() -> str:
    """Build the planner system prompt with real categories from the database."""
    return _PLANNER_SYSTEM_TEMPLATE.format(
        categories=get_categories_for_prompt(),
    )


def _format_history(history: list[dict]) -> str:
    """Convert message dicts to a readable conversation string for LLM context."""
    if not history:
        return "None"
    lines: list[str] = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def build_planner_user_prompt(
    user_message: str,
    history: list[dict[str, str]],
    summary: str = "No summary available.",
    last_business_names: list[str] | None = None,
    last_business_ids: list[int] | None = None,
) -> str:
    """Compact context for the structured planner call."""
    last_names_context = ""
    if last_business_names and last_business_ids:
        # Match names and IDs positionally
        names_list = "\n".join(
            f"[ID: {bid}] {name}" 
            for bid, name in zip(last_business_ids, last_business_names)
        )
        last_names_context = f"\nRecent Business ID Dictionary (for exact ID mapping):\n{names_list}\n"

    return f"""
User message:
{user_message}

Recent conversation history:
{_format_history(history)}

Conversation Summary:
{summary}
{last_names_context}
""".strip()


# ---------------------------------------------------------------------------
# Business data formatters — two modes
# ---------------------------------------------------------------------------

def _format_business_summary(bundle: dict, index: int) -> str:
    """
    Brief card for list view — shown during initial business_search results.
    Contact details intentionally omitted; user must ask for them explicitly.
    """
    b = bundle.get("business") or {}
    lines: list[str] = [f"--- Business {index} ---"]

    lines.append(f"Name: {b.get('business_name') or 'N/A'}")
    category = b.get("category_name") or "N/A"
    sub = b.get("sub_category_name")
    lines.append(f"Category: {category}" + (f" / {sub}" if sub else ""))
    lines.append(f"City: {b.get('city') or 'N/A'}")

    if b.get("business_address"):
        lines.append(f"Address: {b['business_address']}")

    if b.get("message"):
        lines.append(f"Description: {str(b['message'])[:500]}")

    # First highlight — services headline only
    for h in (bundle.get("highlights") or [])[:1]:
        if h.get("products_or_services"):
            lines.append(f"Services: {str(h['products_or_services'])[:400]}")

    return "\n".join(lines)


def _format_business_detail(bundle: dict, index: int) -> str:
    """
    Full card for detail view — shown when user asks for more info or contact details.
    Includes all contact values (phone, WhatsApp, email, website, social).
    LLM must share whatever is present here; say 'not available' for missing fields.
    """
    b = bundle.get("business") or {}
    lines: list[str] = [f"--- Business {index} ---"]

    lines.append(f"Name: {b.get('business_name') or 'N/A'}")
    category = b.get("category_name") or "N/A"
    sub = b.get("sub_category_name")
    lines.append(f"Category: {category}" + (f" / {sub}" if sub else ""))
    lines.append(f"City: {b.get('city') or 'N/A'}")

    if b.get("business_address"):
        lines.append(f"Address: {b['business_address']}")
    if b.get("package_status"):
        lines.append(f"Package: {b['package_status']}")
    if b.get("business_model"):
        lines.append(f"Model: {b['business_model']}")

    # Full contact details
    if b.get("mobile_number"):
        lines.append(f"Phone: {b['mobile_number']}")
    if b.get("whatsapp_number"):
        lines.append(f"WhatsApp: {b['whatsapp_number']}")
    if b.get("email"):
        lines.append(f"Email: {b['email']}")
    if b.get("website_url"):
        lines.append(f"Website: {b['website_url']}")
    if b.get("instagram_social_link"):
        lines.append(f"Instagram: {b['instagram_social_link']}")
    if b.get("facebook_social_link"):
        lines.append(f"Facebook: {b['facebook_social_link']}")

    if b.get("message"):
        lines.append(f"Description: {str(b['message'])[:800]}")

    # First highlight block
    for h in (bundle.get("highlights") or [])[:1]:
        if h.get("business_heading"):
            lines.append(f"Heading: {h['business_heading']}")
        if h.get("products_or_services"):
            lines.append(f"Services: {str(h['products_or_services'])[:500]}")

    # Top 3 reviews
    review_lines: list[str] = []
    for r in (bundle.get("reviews") or [])[:3]:
        text = str(r.get("review") or "").strip()[:150]
        if not text:
            continue
        reviewer = r.get("name") or "Customer"
        score = r.get("review_score")
        score_str = f" ({score}/5)" if score is not None else ""
        review_lines.append(f"  • {reviewer}{score_str}: {text}")
    if review_lines:
        lines.append("Reviews:")
        lines.extend(review_lines)

    # Top 3 FAQs
    faq_lines: list[str] = []
    for f in (bundle.get("faqs") or [])[:3]:
        q = str(f.get("question") or "").strip()
        a = str(f.get("answer") or "").strip()[:200]
        if q and a:
            faq_lines.append(f"  Q: {q}\n  A: {a}")
    if faq_lines:
        lines.append("FAQs:")
        lines.extend(faq_lines)

    return "\n".join(lines)


def _format_businesses_for_prompt(businesses: list[dict], detail_mode: bool = False) -> str:
    """
    Format business bundles for the answer prompt.

    detail_mode=False (default): summary cards only — for initial search results.
    detail_mode=True: full cards with contact info — for follow-up / detail requests.
    """
    if not businesses:
        return "No businesses found."

    formatter = _format_business_detail if detail_mode else _format_business_summary
    sections = [formatter(bundle, i) for i, bundle in enumerate(businesses, 1)]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """
You are a memory summarizer for a business-listings chatbot.
Compress the conversation history into a concise summary of the user's preferences, past searches, and intent.
DO NOT include pleasantries. Keep it under 3 sentences.
Example: "User is looking for cheap baby products. Previously searched for a gym in Karachi."
""".strip()

ANSWER_SYSTEM_PROMPT = """
You are a friendly, helpful, but BRUTALLY HONEST business-listings assistant.

Language rule (strictly follow):
- Default language is English. Always reply in English unless the user clearly writes in Roman Urdu or Urdu.
- If the user's message is clearly in Roman Urdu or Urdu script, reply in Roman Urdu.
- Never mix languages within a single reply beyond what the user themselves mixed.
- When in doubt, use English.

HONESTY & FILTERING PROTOCOL (ENTERPRISE STANDARD):
- You will receive a list of fetched businesses. AI search might return a mix of exact matches and loosely related businesses.
- ACT AS A SMART FILTER. If a business is a perfect match (or a synonym like "confectionery" for "sweet shop"), include it normally.
- MINOR LENIENCY: If you cannot find any EXACT matches, or if a business is strongly related (e.g., a bakery or jaggery shop when asking for sweets), YOU MAY include it as a helpful alternative instead of dropping it.
- When providing an alternative, frame it conversationally (e.g., "I couldn't find a dedicated sweet shop, but I did find a great bakery that serves traditional sweets!").
- Do NOT show completely irrelevant businesses (e.g., an auto repair shop when asking for food). Drop completely irrelevant ones silently.
- If ALL provided businesses are completely irrelevant to even the broad category, honestly apologize and say you couldn't find any matches.

MULTI-INTENT HANDLING:
- If the user asks for two completely different things in one message (e.g., "gyms in Lahore and sweet shops in Karachi"), the system will only search for the FIRST one.
- In your answer, provide the results for the first request, and politely explain that you can only search for one category at a time. Ask them if they'd like you to search for the second item next!

Tone & Style (CRITICAL FOR NATURAL FEEL):
- You MUST sound like a warm, extremely polite, and friendly human expert. Speak as if you are a helpful assistant chatting with a friend. No robotic, stiff, or overly formal phrases.
- Open with engaging, conversational hooks. Example: "I'd be happy to help you with that! I took a look at our listings and found a couple of great options..." or "Sure thing! Let me pull up those details for you."
- Use smooth, natural transitions (e.g., "Here's what I found:", "By the way...", "If you'd like...").
- Be empathetic if no exact data is found: "I'm really sorry, but it looks like we don't have exactly what you're looking for right now. However..."
- Always end with a warm, friendly follow-up question (e.g., "Would you like the contact number for any of these?", "Should I look for anything else?").

LIST MODE (when multiple businesses are returned from a search):
- Show a brief summary card for each relevant business: name, category, city, description, services.
- Do NOT include phone numbers, WhatsApp, email, website, or social links in list view.
- After the list, add one friendly line inviting the user to ask for more.

DETAIL MODE (when the user asked for more info or contact about a specific business):
- Present the details in a readable, conversational flow, but keep the actual contact info cleanly formatted.
- Share all available contact details: phone, WhatsApp, email, website, social links.
- If a contact field is not present in the data, just omit it naturally or say it's not available if explicitly asked.
- Include reviews, FAQs, and services if relevant.

CHAT MODE (when the user asks a conversational question or compares businesses from history):
- The `Fetched business data` might be empty. This is expected!
- Look at the `Recent conversation history` to see the businesses you previously recommended.
- Answer the user's question naturally based on the history. E.g., if they ask "Which is best for cakes?", recommend the one that mentioned cakes in the history.
- Be highly conversational, opinionated (but helpful), and context-aware.

General rules:
- Use ONLY the provided business data. Never invent anything.
- Keep it concise and skimmable; do not pad with generic filler.
""".strip()


def build_answer_user_prompt(
    user_message: str,
    history: list[dict[str, str]],
    businesses: list[dict],
    plan: dict,
    detail_mode: bool = False,
) -> str:
    """Prompt for final answer generation from fetched Postgres bundles."""
    if plan.get("action") == "chat":
        mode_hint = "CHAT MODE: User is asking a conversational question. Answer naturally using history."
    elif detail_mode:
        mode_hint = "DETAIL MODE: User asked for more info or contact details about a specific business."
    else:
        mode_hint = "LIST MODE: Show summary cards only. Do NOT show contact details. End with an invitation to ask for more."
    return f"""
User message:
{user_message}

Response mode: {mode_hint}

Plan:
{plan}

Recent conversation history:
{_format_history(history[-6:])}

Fetched business data:
{_format_businesses_for_prompt(businesses, detail_mode=detail_mode)}
""".strip()
