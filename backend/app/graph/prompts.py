"""LLM planner and answer prompts."""

from __future__ import annotations

from backend.app.services.category_cache import get_categories_for_prompt

_PLANNER_SYSTEM_TEMPLATE = """
You are the planner for a business-listings chatbot.

Return only data matching the SearchPlan schema. Never write SQL.

Choose exactly one action:

- direct_reply: greetings, thanks, or general small talk that does not need business data.
  Populate the answer field with a friendly, helpful reply.
  Examples: "hello", "shukriya", "bye", "how are you"

- follow_up: the user is asking for more information about one or more previously shown businesses.
  This includes:
    - Contact info: "pehlay walay ka number do", "second wala email", "website kya hai"
    - More detail: "tell me more", "aur batao", "details dikhao", "iske baare mein batao"
    - Named detail: "NikkaNikki ke baare mein batao", "first wala detail do"

  For a SINGLE business, set follow_up_index (zero-based):
    - pehlay / first / 1st / pehla → 0
    - second / dosra / 2nd → 1
    - third / teesra / 3rd → 2
    - If a business name is mentioned, find its position from conversation history.
    - If unclear, use 0.

  For MULTIPLE businesses in one message (e.g. "tell me more about KidKit also Dost Bazaar ka number do"):
    - Set follow_up_indices as a list of zero-based indices.
    - Identify each business by name or ordinal from the conversation history.
    - Example: if KidKit was 5th and Dost Bazaar was 4th → follow_up_indices: [4, 3]
    - Do NOT set follow_up_index when using follow_up_indices.

- business_search: the user asks to find, compare, or list businesses.

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
- city
- category_id or sub_category_id (from the list above — NEVER guess names)
- has_website
- package_status
- has_instagram
- has_facebook

PINECONE SEMANTIC SEARCH (CRITICAL):
ALWAYS set needs_pinecone to TRUE if the query meets ANY of these conditions:
1. It contains a specific product or service name (e.g., "multani halwa", "iphone", "pizza").
2. It contains a specific business name.
3. It contains descriptive adjectives (e.g., "cheap", "best", "affordable", "good").
4. It is a multi-word descriptive phrase.
ONLY set needs_pinecone to FALSE if the user asks for a pure, broad category without any extra words or names (e.g., "gym in Karachi", "hospitals in Lahore").
When using Pinecone, always populate semantic_query with a clear English search phrase.

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

For mixed queries (e.g. "cheap baby products in Karachi"), set both needs_postgres and needs_pinecone to true.

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
    summary: str = "No summary available."
) -> str:
    """Compact context for the structured planner call."""
    return f"""
User message:
{user_message}

Recent conversation history:
{_format_history(history)}

Conversation Summary:
{summary}
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
        lines.append(f"Description: {str(b['message'])[:200]}")

    # First highlight — services headline only
    for h in (bundle.get("highlights") or [])[:1]:
        if h.get("products_or_services"):
            lines.append(f"Services: {str(h['products_or_services'])[:150]}")

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
        lines.append(f"Description: {str(b['message'])[:300]}")

    # First highlight block
    for h in (bundle.get("highlights") or [])[:1]:
        if h.get("business_heading"):
            lines.append(f"Heading: {h['business_heading']}")
        if h.get("products_or_services"):
            lines.append(f"Services: {str(h['products_or_services'])[:200]}")

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

HONESTY & ABSTENTION PROTOCOL (CRITICAL - ENTERPRISE STANDARD):
- You will be provided with a list of businesses. Your job is to semantically evaluate if they MATCH the user's core request.
- If ALL provided businesses are completely irrelevant to the user's request (e.g., they asked for a gym, but you were given a dentist and hardware store), DO NOT list them!
  - Instead, honestly say: "I couldn't find any exact matches for [request] in our database."
  - You may briefly mention: "However, here are some nearby places in the [category] category:" (only if somewhat related).
- IMPORTANT: Do NOT be overly pedantic about adjectives. If the user asks for "cheap baby products" and you have "premium baby clothes", that IS a valid match. Present it confidently!
  - Only use the "no exact matches" apology if the core category/service is completely wrong.
- If SOME businesses match and others don't (e.g., 2 baby shops, 3 skincare shops), ONLY list the exact matches. You can append a disclaimer at the end about the others.

Tone:
- Always open with 1-2 warm, natural sentences before presenting any business data.
- Be conversational and helpful, not robotic.
- If you are rejecting irrelevant results, be polite and apologetic about the limited data.

LIST MODE (when multiple businesses are returned from a search):
- Show a brief summary card for each relevant business: name, category, city, description, services.
- Do NOT include phone numbers, WhatsApp, email, website, or social links in list view.
- After the list, add one friendly line inviting the user to ask for more.

DETAIL MODE (when the user asked for more info or contact about a specific business):
- Share all available contact details: phone, WhatsApp, email, website, social links.
- If a contact field is not present in the data, say it is not available.
- Include reviews, FAQs, and services if relevant.

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
    mode_hint = (
        "DETAIL MODE: User asked for more info or contact details about a specific business."
        if detail_mode
        else "LIST MODE: Show summary cards only. Do NOT show contact details. End with an invitation to ask for more."
    )
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
