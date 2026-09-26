"""
bot.py -- magicpin AI Challenge: Vera Merchant AI Assistant
===========================================================
Composes context-aware WhatsApp messages for merchants using Grok (xAI).

Architecture:
  1. Trigger routing  - maps trigger.kind to a named prompt strategy
  2. Context summaries - condenses each JSON context to prompt-friendly text
  3. Prompt composer  - system prompt + user prompt with compulsion-lever guidance
  4. Output validator - post-LLM checks: CTA shape, send_as, suppression_key
  5. compose()        - main entry point; deterministic (temperature=0)

LLM: Google Gemini via OpenAI-compatible endpoint
API key: set GEMINI_API_KEY environment variable before running
  Get free key at: https://aistudio.google.com/apikey
"""
from __future__ import annotations

import os
import json
import re
import pathlib
from typing import Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CLIENT & PROVIDER CONFIGURATION (Grok or Gemini)
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None
_active_model: str = "gemini-3.5-flash-lite"


def _get_client_and_model() -> tuple[OpenAI, str]:
    global _client, _active_model
    if _client is None:
        grok_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if grok_key:
            _client = OpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1",
            )
            _active_model = os.environ.get("GROK_MODEL", "grok-3-mini")
        elif gemini_key:
            _client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            _active_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        else:
            raise EnvironmentError(
                "Neither GROK_API_KEY nor GEMINI_API_KEY is set.\n"
                "Please open .env and enter your key."
            )
    return _client, _active_model


# ---------------------------------------------------------------------------
# TRIGGER ROUTING  -- trigger.kind -> strategy name
# ---------------------------------------------------------------------------
TRIGGER_STRATEGIES: dict = {
    "research_digest": "research_insight",
    "regulation_change": "compliance_urgency",
    "category_trend_movement": "trend_opportunity",
    "festival_upcoming": "festival_campaign",
    "weather_heatwave": "local_event",
    "local_news_event": "local_event",
    "competitor_opened": "competitive_alert",
    "category_research_digest_release": "research_insight",
    "perf_spike": "capitalize_momentum",
    "perf_dip": "recovery_action",
    "milestone_reached": "milestone_celebration",
    "dormant_with_vera": "re_engagement",
    "review_theme_emerged": "review_response",
    "renewal_due": "renewal_urgency",
    "curious_ask_due": "curiosity_ask",
    "scheduled_recurring": "curiosity_ask",
    "unverified_gbp": "profile_fix",
    "recall_due": "patient_recall",
    "customer_lapsed_soft": "lapsed_reactivation",
    "appointment_tomorrow": "appointment_reminder",
    "chronic_refill_due": "refill_reminder",
    "trial_followup": "trial_followup",
    "wedding_package_followup": "wedding_followup",
    "winback": "winback",
}

STRATEGY_INSTRUCTIONS: dict = {
    "research_insight": (
        "Lead with the specific research finding: study size, % improvement, source citation. "
        "Connect it explicitly to THIS merchant's patient/customer cohort using data from the context. "
        "Offer to pull the full item AND draft a patient-education WhatsApp they can reshare. "
        "Tone: peer-clinical colleague. No binary CTA -- use open-ended curiosity question."
    ),
    "compliance_urgency": (
        "Open with the specific regulatory change name and its exact deadline date. "
        "State precisely what the merchant must do. "
        "Use loss-aversion: what non-compliance costs or risks. "
        "Single binary CTA: Reply YES to get a step-by-step compliance checklist."
    ),
    "trend_opportunity": (
        "Anchor on the specific trend: search query name plus % YoY growth. "
        "Tie it to a specific service+price from the merchant's own catalog. "
        "Curiosity CTA: Want to see how peers in your city are positioning this?"
    ),
    "festival_campaign": (
        "Name the festival and exact days remaining. "
        "Propose a specific service+price offer from the merchant's own catalog. "
        "Effort externalization: I have already drafted the post -- just say go. "
        "Single YES/STOP CTA."
    ),
    "local_event": (
        "Name the local event or weather fact precisely. "
        "Explain specifically how it affects footfall or demand for this category. "
        "Propose one quick actionable step. Keep brief."
    ),
    "competitive_alert": (
        "Factually state a competitor has opened nearby. "
        "Anchor on this merchant's specific advantage: rating, review count, or active offer. "
        "Loss aversion: customers are comparing on Google right now. "
        "CTA: Want me to push your best post to the top of your GBP today?"
    ),
    "capitalize_momentum": (
        "Open with the specific metric that spiked, e.g., views +28%. "
        "Frame as a conversion window: browsers are becoming callers right now. "
        "Propose a specific offer or profile action to capitalize on it. "
        "Single YES/STOP CTA."
    ),
    "recovery_action": (
        "Open with the exact metric drop: calls -X% week-on-week vs baseline of Y. "
        "Propose one specific probable cause based on the merchant's signals. "
        "Propose a specific actionable fix. "
        "Single YES/STOP CTA."
    ),
    "milestone_celebration": (
        "Celebrate the specific milestone with the exact number. "
        "Add peer context from peer_stats showing where this puts them. "
        "Propose the next concrete step to capitalize. "
        "Keep warm and brief. Open question CTA or none."
    ),
    "re_engagement": (
        "Acknowledge the gap gently without being naggy. "
        "Use reciprocity: I noticed something about your account you would want to know. "
        "Share one specific insight from the data: stale posts, missed opportunity, peer comparison. "
        "Low-friction CTA: one simple question or YES to resume."
    ),
    "review_response": (
        "Reference the specific review theme by name and occurrence count. "
        "Give one concrete suggestion: a policy change or profile response. "
        "Offer to draft a response template. "
        "Peer tone -- helpful colleague, not alarm bells."
    ),
    "renewal_urgency": (
        "State days remaining clearly at the start. "
        "Quantify what they lose if lapsed: visibility, offers, contact data. "
        "Loss aversion is the core lever. "
        "Single YES/STOP CTA."
    ),
    "curiosity_ask": (
        "Ask exactly ONE specific curious question about their business this week. "
        "Make it feel like market research they would genuinely want to share. "
        "No CTA beyond the question itself. Keep it conversational and brief."
    ),
    "profile_fix": (
        "Name the specific profile gap: unverified, missing description, no recent posts. "
        "Quantify the cost with a peer stat from the context. "
        "Effort externalization: I can fix this in 2 minutes -- just say go. "
        "Single YES CTA."
    ),
    "patient_recall": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Address patient by first name. "
        "State the service due and months since last visit (calculate from the dates). "
        "Offer the specific available slots from the trigger payload. "
        "Use Hindi-English code-mix if patient language_pref is hi-en mix or hi. "
        "Multi-slot booking CTA is allowed: Reply 1 for X, 2 for Y."
    ),
    "lapsed_reactivation": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Address customer by first name. "
        "Reference their last service and how many months ago it was. "
        "Offer one relevant service+price from catalog. "
        "Single gentle CTA."
    ),
    "appointment_reminder": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Confirm appointment: date, time, service. "
        "Add one helpful practical tip for the service. "
        "Simple confirmation CTA."
    ),
    "refill_reminder": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Reference the specific medication or service type if known from context. "
        "Make reordering friction-free: Same as last time? Reply YES. "
        "Single YES CTA."
    ),
    "trial_followup": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Reference what the trial or first session covered. "
        "Ask how it went -- curiosity plus reciprocity. "
        "Offer next step booking."
    ),
    "wedding_followup": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Reference the wedding date and the trial session date from payload. "
        "Propose the next step in bridal prep with specific timing. "
        "Gentle urgency: the window is now open."
    ),
    "winback": (
        "IMPORTANT: send_as must be merchant_on_behalf. "
        "Reference last visit date as months ago. "
        "Loss aversion plus a specific offer to return. "
        "One simple CTA."
    ),
}


# ---------------------------------------------------------------------------
# CONTEXT SUMMARIZERS
# ---------------------------------------------------------------------------

def _summarize_category(cat: dict) -> str:
    voice = cat.get("voice", {})
    peer = cat.get("peer_stats", {})
    offers = cat.get("offer_catalog", [])[:4]
    digest = cat.get("digest", [])[:3]
    seasonal = cat.get("seasonal_beats", [])[:2]
    trends = cat.get("trend_signals", [])[:2]

    offer_list = " | ".join(o.get("title", "") for o in offers)
    digest_lines = []
    for d in digest:
        effect = d.get("effect_size", d.get("effect", d.get("improvement", "?")))
        digest_lines.append(
            "  [" + d.get("source", "?") + "] " + d.get("title", "")
            + " n=" + str(d.get("trial_n", "?")) + " effect=" + str(effect)
        )
    digest_str = "\n".join(digest_lines) or "  none"
    seasonal_str = " | ".join(
        s.get("month_range", s.get("month", "?")) + ": " + s.get("note", "")
        for s in seasonal
    ) or "none"
    trend_str = " | ".join(
        t.get("query", "?") + " +" + str(int(t.get("delta_yoy", 0) * 100)) + "% YoY"
        for t in trends
    ) or "none"

    return "\n".join([
        "CATEGORY: " + cat.get("display_name", cat.get("slug", "?")),
        "Voice: tone=" + voice.get("tone", "?") + " register=" + voice.get("register", "?") + " code_mix=" + voice.get("code_mix", "?"),
        "Taboo words: " + ", ".join(voice.get("vocab_taboo", [])[:5]),
        "Peer stats: avg_rating=" + str(peer.get("avg_rating", "?")) + " avg_CTR=" + str(peer.get("avg_ctr", "?")) + " avg_views_30d=" + str(peer.get("avg_views_30d", "?")) + " avg_calls_30d=" + str(peer.get("avg_calls_30d", "?")),
        "Catalog offers: " + offer_list,
        "Seasonal beats: " + seasonal_str,
        "Trend signals: " + trend_str,
        "Research digest:\n" + digest_str,
    ])


def _summarize_merchant(m: dict) -> str:
    identity = m.get("identity", {})
    sub = m.get("subscription", {})
    perf = m.get("performance", {})
    delta = perf.get("delta_7d", {})
    offers = m.get("offers", [])
    history = m.get("conversation_history", [])[-3:]
    signals = m.get("signals", [])
    reviews = m.get("review_themes", [])
    agg = m.get("customer_aggregate", {})

    active_offers = [o for o in offers if o.get("status") == "active"]
    offer_str = " | ".join(o.get("title", "") for o in active_offers) or "none"
    history_lines = ["  [" + h.get("from", "?") + "]: " + h.get("body", "") for h in history] or ["  (no history)"]
    signals_str = ", ".join(str(s) for s in signals) or "none"
    review_parts = [
        r.get("theme", "?") + " (" + r.get("sentiment", "?") + ", " + str(r.get("occurrences_30d", 0)) + "x): \"" + r.get("common_quote", "") + "\""
        for r in reviews
    ]
    review_str = " | ".join(review_parts) or "none"
    views_delta = delta.get("views_pct", 0) * 100
    calls_delta = delta.get("calls_pct", 0) * 100

    lines = [
        "MERCHANT: " + identity.get("name", "?") + " | " + identity.get("locality", "?") + ", " + identity.get("city", "?"),
        "Owner: " + identity.get("owner_first_name", "?") + " | Verified: " + str(identity.get("verified", "?")) + " | Est: " + str(identity.get("established_year", "?")),
        "Languages: " + ", ".join(identity.get("languages", ["en"])),
        "Subscription: " + sub.get("status", "?") + " (" + sub.get("plan", "?") + ") days_remaining=" + str(sub.get("days_remaining", "?")) + " days_since_expiry=" + str(sub.get("days_since_expiry", "N/A")),
        "Performance (30d): views=" + str(perf.get("views", "?")) + " calls=" + str(perf.get("calls", "?")) + " directions=" + str(perf.get("directions", "?")) + " CTR=" + str(perf.get("ctr", "?")) + " leads=" + str(perf.get("leads", "?")),
        "7d delta: views " + (("+" if views_delta >= 0 else "") + str(int(views_delta)) + "%") + " calls " + (("+" if calls_delta >= 0 else "") + str(int(calls_delta)) + "%"),
        "Active offers: " + offer_str,
        "Customer aggregate: total_ytd=" + str(agg.get("total_unique_ytd", "?")) + " lapsed_180d=" + str(agg.get("lapsed_180d_plus", "N/A")) + " retention_6mo=" + str(agg.get("retention_6mo_pct", "N/A")),
        "Signals: " + signals_str,
        "Review themes: " + review_str,
        "Conversation history:",
    ] + history_lines
    return "\n".join(lines)


def _summarize_trigger(t: dict) -> str:
    payload_str = json.dumps(t.get("payload", {}), ensure_ascii=False, indent=2)
    return "\n".join([
        "TRIGGER: " + t.get("id", "?"),
        "Kind=" + t.get("kind", "?") + " | Scope=" + t.get("scope", "?") + " | Source=" + t.get("source", "?") + " | Urgency=" + str(t.get("urgency", "?")) + "/5",
        "Expires: " + t.get("expires_at", "?"),
        "Payload:\n" + payload_str,
    ])


def _summarize_customer(c: dict) -> str:
    if not c:
        return ""
    identity = c.get("identity", {})
    rel = c.get("relationship", {})
    prefs = c.get("preferences", {})
    consent = c.get("consent", {})
    services = rel.get("services_received", [])
    return "\n".join([
        "CUSTOMER: " + identity.get("name", "?") + " | lang_pref=" + identity.get("language_pref", "en") + " | age_band=" + identity.get("age_band", "?"),
        "State: " + c.get("state", "?") + " | visits_total=" + str(rel.get("visits_total", "?")) + " | last_visit=" + str(rel.get("last_visit", "?")) + " | first_visit=" + str(rel.get("first_visit", "?")),
        "Services received: " + (", ".join(services) if services else "not listed"),
        "Preferences: reminder_opt_in=" + str(prefs.get("reminder_opt_in", "?")),
        "Consent scope: " + ", ".join(consent.get("scope", ["none"])),
    ])


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Vera -- magicpin's merchant AI assistant on WhatsApp.\n"
    "You compose highly personalized, specific, compulsion-driven messages for Indian merchants and their customers.\n\n"
    "MANDATORY RULES (violating any = bad score):\n"
    "1. SPECIFICITY: Always anchor on a verifiable fact from the context: number, date, source, stat. Never generic.\n"
    "2. VOICE MATCH: Match category tone exactly.\n"
    "   Dentists=peer-clinical. Salons=warm-friendly. Restaurants=casual-enthusiastic. Gyms=energetic. Pharmacies=trustworthy-practical.\n"
    "3. MERCHANT FIT: Use THIS merchant's actual numbers, offers, and conversation history.\n"
    "4. SINGLE CTA: One action only. Binary (YES/STOP) for action triggers. Open question for info triggers. None for pure curiosity.\n"
    "5. LANGUAGE: Use Hindi-English code-mix naturally for Indian merchants unless they prefer pure English.\n"
    "6. NO FABRICATION: Only data from the given contexts. No invented names, fake research, or fake offers.\n"
    "7. NO ANTI-PATTERNS: No hope-you-are-doing-well, no re-introductions after first message, no multiple CTAs, no AMAZING DEAL hype.\n"
    "8. CONCISE: WhatsApp messages. 3-6 sentences max.\n\n"
    "COMPULSION LEVERS (use 1-2 per message):\n"
    "- Specificity: concrete number, date, stat, source citation\n"
    "- Loss aversion: you are missing X / before this window closes\n"
    "- Social proof: 3 dentists in your locality did Y this month\n"
    "- Effort externalization: I have drafted it -- just say go\n"
    "- Curiosity: want to see who? / want the full list?\n"
    "- Reciprocity: I noticed Y about your account, thought you would want to know\n"
    "- Ask the merchant: what is your most-asked service this week?\n"
    "- Binary commitment: Reply YES / STOP\n\n"
    'OUTPUT FORMAT -- JSON only, no markdown:\n'
    '{\n'
    '  "body": "<WhatsApp message body>",\n'
    '  "cta": "yes_stop" or "open_ended" or "none",\n'
    '  "send_as": "vera" or "merchant_on_behalf",\n'
    '  "suppression_key": "<from trigger suppression_key field>",\n'
    '  "rationale": "<1-2 sentences: why this message, which lever, what it achieves>"\n'
    '}'
)


# ---------------------------------------------------------------------------
# MAIN COMPOSE FUNCTION
# ---------------------------------------------------------------------------

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Compose a WhatsApp message.

    Args:
        category: CategoryContext dict (from categories/*.json)
        merchant: MerchantContext dict (from merchants/*.json)
        trigger:  TriggerContext dict (from triggers/*.json)
        customer: CustomerContext dict (optional, from customers/*.json)

    Returns:
        dict with keys: body, cta, send_as, suppression_key, rationale
    """
    kind = trigger.get("kind", "")
    strategy = TRIGGER_STRATEGIES.get(kind, "re_engagement")
    strategy_instruction = STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS["re_engagement"])

    cat_summary = _summarize_category(category)
    merchant_summary = _summarize_merchant(merchant)
    trigger_summary = _summarize_trigger(trigger)
    customer_section = _summarize_customer(customer) if customer else "CUSTOMER: None (merchant-facing message)"

    user_prompt = (
        "Compose a WhatsApp message using the context below.\n\n"
        "--- CONTEXT ---\n"
        + cat_summary + "\n\n"
        + merchant_summary + "\n\n"
        + trigger_summary + "\n\n"
        + customer_section + "\n\n"
        + "--- STRATEGY: " + strategy.upper() + " ---\n"
        + strategy_instruction + "\n\n"
        + "Output valid JSON only."
    )

    client, model_name = _get_client_and_model()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {
                "body": raw,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": "JSON parse fallback",
            }

    # Post-processing validation
    if trigger.get("scope") == "customer" and customer:
        result["send_as"] = "merchant_on_behalf"
    elif trigger.get("scope") == "merchant":
        result["send_as"] = "vera"

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", "msg:" + trigger.get("id", "unknown")
        )

    cta = result.get("cta", "")
    if cta not in {"yes_stop", "open_ended", "none"}:
        body = result.get("body", "")
        if re.search(r"reply yes", body, re.IGNORECASE):
            result["cta"] = "yes_stop"
        elif "?" in body:
            result["cta"] = "open_ended"
        else:
            result["cta"] = "none"

    return result


# ---------------------------------------------------------------------------
# DATASET LOADER HELPER
# ---------------------------------------------------------------------------

def load_context(
    dataset_dir: str,
    merchant_id: str,
    trigger_id: str,
    customer_id: Optional[str] = None,
):
    """Load all 4 contexts from the expanded dataset directory."""
    base = pathlib.Path(dataset_dir)

    with open(base / "merchants" / (merchant_id + ".json"), encoding="utf-8") as f:
        merchant = json.load(f)

    with open(base / "categories" / (merchant["category_slug"] + ".json"), encoding="utf-8") as f:
        category = json.load(f)

    with open(base / "triggers" / (trigger_id + ".json"), encoding="utf-8") as f:
        trigger = json.load(f)

    customer = None
    if customer_id:
        with open(base / "customers" / (customer_id + ".json"), encoding="utf-8") as f:
            customer = json.load(f)

    return category, merchant, trigger, customer


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cat, m, t, c = load_context(
        "dataset/expanded",
        "m_001_drmeera_dentist_delhi",
        "trg_001_research_digest_dentists",
    )
    result = compose(cat, m, t, c)
    print(json.dumps(result, indent=2, ensure_ascii=False))
