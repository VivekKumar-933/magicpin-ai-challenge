# Vera Bot — magicpin AI Challenge Submission

**Model**: Grok-3-mini (xAI) via OpenAI-compatible API | **Temperature**: 0 (fully deterministic)

---

## Approach

### 1. Trigger Routing (25 strategies)
Every trigger `kind` maps to a named strategy (e.g., `research_digest` → `research_insight`, `perf_dip` → `recovery_action`). Each strategy has explicit instructions for:
- Which compulsion lever(s) to lead with
- What data to anchor on (from which context field)
- What CTA shape to use

This prevents the bot from giving a "generic nudge" when a specific, high-scoring framing exists.

### 2. Context Summarization
Each of the 4 contexts is condensed to a dense, prompt-friendly text block. Only the most relevant fields are included (top 3 digest items, top 4 offers, last 3 conversation turns). This keeps the prompt ~2K tokens while preserving all decision-relevant information.

### 3. System Prompt with Compulsion Lever List
The system prompt hardcodes all 8 compulsion levers from the brief, with explicit instruction to use 1-2 per message. The mandatory-rules section mirrors the 5 scoring dimensions exactly.

### 4. Post-LLM Validation
After parsing the JSON response:
- `send_as` is forced to `merchant_on_behalf` for customer-scope triggers
- `suppression_key` is filled from the trigger if missing
- `cta` is normalized to `yes_stop | open_ended | none`

### 5. Multi-Turn Handling (`conversation_handlers.py`)
- **Auto-reply detection**: regex patterns + exact-repeat check (same message ≥2x = auto-reply)
- **Intent routing**: `join_intent` → skip re-qualifying, go straight to action; `not_interested` → 1-sentence warm exit
- **Exit logic**: closes conversation after 2 auto-replies or 3 unanswered nudges
- **LLM-backed general turn**: full conversation history passed to Grok for contextual replies

---

## Tradeoffs

| Decision | Why |
|---|---|
| Grok-3-mini over larger models | Fast (< 5s/call), cheap, deterministic at temperature=0; quality sufficient for structured JSON output |
| In-memory state in server.py | Judge doesn't restart between calls; this is simpler and faster than a DB |
| 25 fixed strategies vs. pure LLM routing | More predictable scoring; avoids the LLM choosing a wrong framing |
| Hindi-English mix by default | Most Indian merchants respond better to code-mix; pure Hindi/English is set by language preference in context |

---

## What additional context would have helped most

1. **Merchant's GBP photo count and quality score** — would let the bot make more specific "profile improvement" offers
2. **Real appointment slots** — triggers reference "available_slots" but most generated triggers have placeholder payloads; real slot data would dramatically improve patient-recall and appointment-reminder messages
3. **Historical campaign performance** — knowing which offer types got the most merchant YES responses would let the bot pick offers more intelligently
4. **Merchant's WhatsApp reply rate baseline** — knowing if a merchant typically replies in 1h vs. never would let the bot calibrate urgency

---

## Files

| File | Purpose |
|---|---|
| `bot.py` | Core `compose()` function |
| `conversation_handlers.py` | Multi-turn `respond()` function + auto-reply + intent detection |
| `server.py` | FastAPI HTTP server (5 endpoints) |
| `generate_submission.py` | Generates `submission.jsonl` for all 30 test pairs |
| `submission.jsonl` | 30 pre-generated messages (the actual submission) |
| `dataset/` | Seed data + generator |
| `dataset/expanded/` | Full 50-merchant, 200-customer, 100-trigger dataset |

---

## Running

```powershell
# 1. Set API key
$env:GROK_API_KEY = "xai-..."

# 2. Generate the dataset (already done)
python dataset/generate_dataset.py --out dataset/expanded

# 3. Generate submission.jsonl
python generate_submission.py

# 4. Start the HTTP server for live judge testing
python server.py
# -> http://localhost:8080

# 5. Smoke test compose()
python bot.py
```
