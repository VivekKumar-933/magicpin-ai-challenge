"""
conversation_handlers.py -- magicpin AI Challenge
==================================================
Multi-turn conversation handling for Vera.

Implements:
  - Auto-reply detection (canned WhatsApp Business replies)
  - Intent routing (join intent, action intent, not-interested)
  - Conversation state management
  - Graceful exit logic (after 3 unanswered nudges or explicit STOP)
  - respond() - the main multi-turn entry point
"""
from __future__ import annotations

import json
import re
import os
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI

# ---------------------------------------------------------------------------
# AUTO-REPLY DETECTION
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"(thank you|shukriya|dhanyawad).{0,40}(contact|reaching out|message)",
    r"(automated|automatic).{0,30}(reply|response|assistant)",
    r"(out of office|unavailable|currently away)",
    r"(team tak pahuncha|team ko forward|pahuncha dungi)",
    r"(main ek automated|i am an automated|this is an automated)",
    r"(aapki jaankari ke liye shukriya|information ke liye dhanyawad)",
    r"(we will get back|jald hi sampark karenge|team will contact)",
    r"(business hours.{0,30}between|available from)",
    r"(aapki madad ke liye shukriya).{0,30}(automated|assistant)",
]

EXACT_REPEAT_THRESHOLD = 2


def detect_auto_reply(message: str, message_history: list) -> bool:
    """
    Returns True if the message appears to be a WhatsApp Business auto-reply.
    Checks regex patterns + exact-repeat detection.
    """
    msg_lower = message.lower().strip()

    for pattern in AUTO_REPLY_PATTERNS:
        if re.search(pattern, msg_lower):
            return True

    prev_merchant = [
        h["body"].lower().strip()
        for h in message_history
        if h.get("from") == "merchant"
    ]
    if prev_merchant.count(msg_lower) >= EXACT_REPEAT_THRESHOLD:
        return True

    return False


# ---------------------------------------------------------------------------
# INTENT DETECTION
# ---------------------------------------------------------------------------

JOIN_INTENT_PATTERNS = [
    r"(join|judrna|judna|join karna|register|sign up|signup)",
    r"(main interested|mujhe interest|haan main chahta|yes i want to join)",
    r"(magicpin mein|magicpin pe|platform pe)",
]

ACTION_INTENT_PATTERNS = [
    r"^(yes|haan|ha|bilkul|zaroor|sure|okay|ok)[\s!.]*$",
    r"(go ahead|kar do|update karo|post karo|draft karo|chal|chalo)",
    r"(let.s do it|theek hai|agree|approved)",
]

NOT_INTERESTED_PATTERNS = [
    r"(not interested|interested nahi|nahi chahiye|no thanks)",
    r"^(stop|band karo|mat bhejo|unsubscribe)[\s!.]*$",
    r"(don.t want|nahi chahta|nahi chahti|please stop|abhi nahi)",
]

QUESTION_PATTERNS = [
    r"\?(\s|$)",
    r"\b(kya|what|how|kyun|why|kab|when|kitna|how much|kaisa|kaisi|bataiye|batao)\b",
]


def detect_intent(message: str) -> str:
    """Returns: join_intent | action_intent | not_interested | question | neutral"""
    msg_lower = message.lower().strip()

    for p in NOT_INTERESTED_PATTERNS:
        if re.search(p, msg_lower):
            return "not_interested"

    for p in JOIN_INTENT_PATTERNS:
        if re.search(p, msg_lower):
            return "join_intent"

    for p in ACTION_INTENT_PATTERNS:
        if re.search(p, msg_lower):
            return "action_intent"

    for p in QUESTION_PATTERNS:
        if re.search(p, msg_lower):
            return "question"

    return "neutral"


# ---------------------------------------------------------------------------
# CONVERSATION STATE
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Tracks state for a single merchant conversation."""
    merchant_id: str
    conversation_id: str
    history: list = field(default_factory=list)  # [{from, body, ts}]
    unanswered_nudges: int = 0
    auto_reply_attempts: int = 0
    last_vera_trigger: str = ""
    is_closed: bool = False
    merchant_language: str = "hi-en mix"

    def add_message(self, from_: str, body: str, ts: str = ""):
        self.history.append({"from": from_, "body": body, "ts": ts})
        if from_ == "merchant":
            self.unanswered_nudges = 0  # reset on any merchant reply

    def should_exit(self) -> bool:
        return self.is_closed or self.unanswered_nudges >= 3


# ---------------------------------------------------------------------------
# MULTI-TURN LLM
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _client


MULTI_TURN_SYSTEM = (
    "You are Vera -- magicpin's merchant AI assistant on WhatsApp. "
    "You are in an ongoing conversation with a merchant. "
    "Rules:\n"
    "1. If the merchant said YES or showed action intent, move to action mode immediately. Do NOT re-qualify.\n"
    "2. If the merchant asked a question, answer it concisely and specifically using only known facts.\n"
    "3. If the merchant is clearly not interested, give a warm 1-sentence graceful exit. No hard sell.\n"
    "4. If responding after an auto-reply, try once more with a gentle human hook.\n"
    "5. Keep messages to 2-4 sentences. No long paragraphs.\n"
    "6. Use Hindi-English mix naturally if the merchant does so.\n"
    "7. No re-introduction. No generic openers like 'Hope you are doing well'.\n\n"
    'OUTPUT JSON: {"body": "...", "cta": "yes_stop|open_ended|none", "is_exit": false, "rationale": "..."}'
)


def _llm_respond(user_prompt: str, state: ConversationState) -> dict:
    """Internal: call LLM (Grok or Gemini) for multi-turn response."""
    from bot import _get_client_and_model
    client, model_name = _get_client_and_model()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": MULTI_TURN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"body": raw, "cta": "open_ended", "is_exit": False, "rationale": "parse fallback"}

    # Update state
    if result.get("is_exit"):
        state.is_closed = True
    state.add_message("vera", result.get("body", ""))
    state.unanswered_nudges += 1

    return result


# ---------------------------------------------------------------------------
# MAIN RESPOND FUNCTION
# ---------------------------------------------------------------------------

def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Given the conversation state + the merchant's latest message, produce the next Vera reply.

    Args:
        state: ConversationState tracking history and flags
        merchant_message: the merchant's incoming WhatsApp message

    Returns:
        dict: body, cta, is_exit, rationale
    """
    if state.is_closed:
        return {"body": "", "cta": "none", "is_exit": True, "rationale": "Conversation already closed."}

    # Detect signals
    is_auto = detect_auto_reply(merchant_message, state.history)
    intent = detect_intent(merchant_message)

    # Record merchant message
    state.add_message("merchant", merchant_message)

    # Handle auto-reply
    if is_auto:
        state.auto_reply_attempts += 1
        if state.auto_reply_attempts >= 2:
            state.is_closed = True
            reply_body = (
                "Koi baat nahi -- main owner/manager se directly connect kar lungi. "
                "Aapka business accha chal raha hai! "
            )
            state.add_message("vera", reply_body)
            return {
                "body": reply_body,
                "cta": "none",
                "is_exit": True,
                "rationale": "Auto-reply detected twice; graceful exit to avoid burning turns.",
            }
        # First auto-reply: try to reach the real person once
        reply_body = (
            "Lagta hai yeh ek automated reply hai. "
            "Agar aap khud baat kar sakein to ek second lein -- "
            "ek useful update share karna tha apke account ke baare mein."
        )
        state.add_message("vera", reply_body)
        state.unanswered_nudges += 1
        return {
            "body": reply_body,
            "cta": "open_ended",
            "is_exit": False,
            "rationale": "Auto-reply detected once; attempting to reach the real merchant.",
        }

    # Handle explicit not-interested
    if intent == "not_interested":
        state.is_closed = True
        reply_body = (
            "Bilkul samajh gayi -- koi problem nahi. "
            "Jab bhi kuch chahiye, main yahan hoon. Take care! "
        )
        state.add_message("vera", reply_body)
        return {
            "body": reply_body,
            "cta": "none",
            "is_exit": True,
            "rationale": "Merchant signaled not interested; warm graceful exit.",
        }

    # Handle join intent -- route to action IMMEDIATELY, no re-qualifying
    if intent == "join_intent":
        conversation_str = "\n".join(
            "[" + h["from"] + "]: " + h["body"] for h in state.history[-5:]
        )
        user_prompt = (
            "The merchant just expressed intent to JOIN magicpin.\n"
            "Do NOT ask qualifying questions. Route to action immediately.\n"
            "Tell them the ONE concrete next step to get onboarded. Warm and specific. 2-3 sentences.\n\n"
            "Conversation so far:\n" + conversation_str + "\n\n"
            "Merchant just said: " + merchant_message + "\n\n"
            "Output JSON only."
        )
        result = _llm_respond(user_prompt, state)
        result["rationale"] = "Join intent detected; routing to onboarding action immediately without re-qualifying."
        return result

    # General: build LLM prompt from conversation history
    conversation_str = "\n".join(
        "[" + h["from"] + "]: " + h["body"] for h in state.history[-6:]
    )
    user_prompt = (
        "Continue this WhatsApp conversation as Vera.\n\n"
        "Conversation so far:\n" + conversation_str + "\n\n"
        "Merchant just said: " + merchant_message + "\n"
        "Detected intent: " + intent + "\n"
        "Auto-reply attempts so far: " + str(state.auto_reply_attempts) + "\n"
        "Unanswered nudges so far: " + str(state.unanswered_nudges) + "\n\n"
        "Reply as Vera. Output JSON only."
    )
    return _llm_respond(user_prompt, state)
