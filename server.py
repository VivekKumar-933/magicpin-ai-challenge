"""
server.py -- magicpin AI Challenge: Vera Bot HTTP Server
=========================================================
FastAPI server implementing the 5 required judge endpoints:

  GET  /v1/healthz          - health check
  GET  /v1/metadata         - bot metadata
  POST /v1/context          - receive context pushes (category/merchant/customer/trigger)
  POST /v1/tick             - periodic wake-up; bot sends proactive messages
  POST /v1/conversation/{id} - receive a merchant reply; bot responds

Run:
  $env:GROK_API_KEY = 'xai-...'
  python server.py

Or with uvicorn directly:
  uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import os
import uuid
import json
import time
from datetime import datetime, timezone
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bot import compose
from conversation_handlers import ConversationState, respond

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(title="Vera Bot", version="1.0.0")

START_TIME = time.time()

# In-memory stores (fine for the contest -- judge doesn't restart between calls)
_contexts: dict[str, dict] = {}       # context_id -> {version, payload, scope, stored_at}
_conversations: dict[str, ConversationState] = {}  # conversation_id -> state
_sent_messages: list[dict] = []        # log of all messages the bot sent


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_context(scope: str, context_id: str) -> Optional[dict]:
    key = scope + ":" + context_id
    entry = _contexts.get(key)
    return entry["payload"] if entry else None


def _find_active_triggers() -> list[dict]:
    """Return all trigger payloads currently stored."""
    result = []
    for key, entry in _contexts.items():
        if key.startswith("trigger:"):
            result.append(entry)
    return result


# ---------------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# ---------------------------------------------------------------------------

class ContextPush(BaseModel):
    scope: str  # category | merchant | customer | trigger
    context_id: str
    version: int = 1
    payload: dict
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ConversationMessage(BaseModel):
    conversation_id: str
    from_: str = Field(alias="from")  # "merchant" | "customer"
    body: str
    ts: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# ENDPOINT 1: Health Check
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
def healthz():
    categories = sum(1 for k in _contexts if k.startswith("category:"))
    merchants = sum(1 for k in _contexts if k.startswith("merchant:"))
    customers = sum(1 for k in _contexts if k.startswith("customer:"))
    triggers = sum(1 for k in _contexts if k.startswith("trigger:"))
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "contexts_loaded": {
            "category": categories,
            "merchant": merchants,
            "customer": customers,
            "trigger": triggers,
        },
    }


# ---------------------------------------------------------------------------
# ENDPOINT 2: Metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera Bot",
        "team_members": ["Vivek Kumar"],
        "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "approach": (
            "Trigger-routed prompt composer with 25 strategy variants. "
            "Context summarization + compulsion-lever instructions. "
            "Post-LLM validation. Multi-turn with auto-reply detection."
        ),
        "version": "1.0.0",
        "submitted_at": "2026-09-26T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# ENDPOINT 3: Context Push
# ---------------------------------------------------------------------------

@app.post("/v1/context")
def push_context(body: ContextPush):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": "scope must be one of " + str(valid_scopes)},
        )

    key = body.scope + ":" + body.context_id
    existing = _contexts.get(key)

    # Idempotent: same version = no-op
    if existing and existing["version"] == body.version:
        return {"accepted": True, "ack_id": "ack_noop_" + str(body.version), "stored_at": existing["stored_at"]}

    # Stale version: reject
    if existing and existing["version"] > body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": existing["version"]},
        )

    # Store/update
    stored_at = _now_iso()
    _contexts[key] = {
        "scope": body.scope,
        "context_id": body.context_id,
        "version": body.version,
        "payload": body.payload,
        "stored_at": stored_at,
    }

    ack_id = "ack_" + uuid.uuid4().hex[:8]
    return {"accepted": True, "ack_id": ack_id, "stored_at": stored_at}


# ---------------------------------------------------------------------------
# ENDPOINT 4: Tick -- bot decides whether to send proactive messages
# ---------------------------------------------------------------------------

@app.post("/v1/tick")
def tick(body: TickRequest):
    actions = []

    for trigger_id in body.available_triggers:
        trigger_entry = _contexts.get("trigger:" + trigger_id)
        if not trigger_entry:
            continue

        trigger = trigger_entry["payload"]
        merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
        customer_id = trigger.get("customer_id")

        if not merchant_id:
            continue

        # Load contexts
        merchant = _get_context("merchant", merchant_id)
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug", "")
        category = _get_context("category", cat_slug)
        if not category:
            continue

        customer = _get_context("customer", customer_id) if customer_id else None

        # Compose message
        try:
            result = compose(category, merchant, trigger, customer)
        except Exception as exc:
            # Log and skip -- don't crash the tick
            continue

        conv_id = "conv_" + trigger_id[-12:]
        send_as = result.get("send_as", "vera")

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "body": result.get("body", ""),
            "cta": result.get("cta", "none"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale": result.get("rationale", ""),
        }
        actions.append(action)
        _sent_messages.append(action)

        # Initialize conversation state for multi-turn
        if conv_id not in _conversations:
            _conversations[conv_id] = ConversationState(
                merchant_id=merchant_id,
                conversation_id=conv_id,
                last_vera_trigger=trigger_id,
            )
        state = _conversations[conv_id]
        state.add_message("vera", result.get("body", ""), body.now)

    return {"actions": actions, "tick_at": body.now}


# ---------------------------------------------------------------------------
# ENDPOINT 5: Conversation -- receive merchant reply, send Vera response
# ---------------------------------------------------------------------------

@app.post("/v1/conversation/{conversation_id}")
def conversation(conversation_id: str, body: dict):
    merchant_message = body.get("body", body.get("message", ""))
    ts = body.get("ts", _now_iso())

    if conversation_id not in _conversations:
        # New conversation -- create state
        _conversations[conversation_id] = ConversationState(
            merchant_id=body.get("merchant_id", "unknown"),
            conversation_id=conversation_id,
        )

    state = _conversations[conversation_id]

    if state.should_exit():
        return {"body": "", "cta": "none", "is_exit": True, "conversation_id": conversation_id}

    result = respond(state, merchant_message)
    result["conversation_id"] = conversation_id
    _sent_messages.append({**result, "conversation_id": conversation_id, "ts": ts})

    return result


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
