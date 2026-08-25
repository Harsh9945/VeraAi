"""
The composer: turns (category, merchant, trigger, customer?) into a message.

Design notes (see README for the full rationale):
  - One shared system prompt encodes the rubric (specificity, category fit,
    merchant fit, trigger relevance, engagement compulsion) as constraints,
    not just guidance — the model is told exactly what loses points.
  - A small per-trigger-kind "framing hint" is injected, because a
    research_digest message and a recall_due message want different shapes
    (source-citation framing vs. slot-offering framing) even though they
    share the same 4-context input shape.
  - temperature=0 for determinism (the brief requires deterministic output
    given the same inputs for /v1/tick's composer; /v1/reply is allowed to
    vary turn to turn since it's genuinely conversational).
  - Output is forced JSON (via prompt + parsing) matching the exact schema
    the harness expects: body, cta, send_as, suppression_key, rationale.
"""

import json
import os
import re
from typing import Any

from anthropic import Anthropic

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy-key"))
    return _client


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        import urllib.request
        import json
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        full_text = f"{system_prompt}\n\n{user_prompt}"
        body = json.dumps({
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 8000,
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                last_error = e
                import time
                time.sleep(1.0)
        raise last_error
    else:
        client = _get_client()
        model_name = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        resp = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text


MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the composer for Vera, magicpin's merchant-AI assistant. \
You write ONE outbound WhatsApp message per call, from structured context. You are being \
scored on a strict rubric — internalize it as hard constraints, not suggestions:

1. DECISION QUALITY: Do not repeat every available fact. Choose the single most compelling signal (combining trigger, merchant state, and category fit) that should drive the next message.
2. SPECIFICITY: Anchor on verifiable facts—real numbers, offers, dates, and local facts from the input. Never invent a statistic, a research citation, a competitor name, or an offer that isn't in the provided data. Fabrication is a disqualifying failure.
3. CATEGORY FIT: Keep tone true to the business type: clinical/peer for doctors/dentists, warm/practical for salons, operator-to-operator for restaurants, trustworthy/precise for pharmacies. Strictly respect all listed category taboo words—never use them.
4. MERCHANT FIT: Personalize using the merchant's actual name, performance numbers, offers from catalog, and prior conversation history. Honor language preference (devanagari-free Latin Hinglish is preferred for Indian merchants).
5. ENGAGEMENT COMPULSION: Give one strong reason to reply now with a short, low-friction, low-effort next action. Prefer reciprocity ("I've drafted X..."), Cialdini leverage (loss aversion, curiosity), or asking the merchant a direct question.

HARD RULES:
- Exactly ONE call-to-action. Never offer multiple branching choices ("Reply YES for X, NO for Y").
- No preambles ("I hope you're doing well"). Don't re-introduce yourself if there's prior \
conversation history.
- Never repeat a message verbatim that appears in the conversation history you were given.
- Keep it concise — WhatsApp message length, not an email.
- If sending on behalf of the merchant to their customer, never make unverified medical/service \
claims and respect the category's customer-facing taboos strictly.
- Always cite specific sources (e.g., "JIDA Oct 2026, p.14", "DCI circular", batch numbers) when citing research or compliance data.
- Always address the merchant/owner by their first name (e.g., "Suresh", "Lakshmi") if found in the context (e.g., owner_first_name or similar).
- For customer-facing messages, honor the customer's specific language preference (e.g. Hinglish mix, Namaste salutations for seniors) and relationship state.
- Use precise domain-specific terminology (e.g., "covers", "AOV", "sub-potency", "ad spend", "conversion", "caries") to demonstrate deep category fit.
- Propose actionable, low-friction next steps (e.g. "Want me to draft X? Live in 10 min", "Reply YES — no commitment").
- Act as a strategic partner: offer logical judgment (e.g. recommending against a match-night promo on a busy Saturday, or pausing spend during a seasonal lull) rather than mindless copy-pasting.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{
  "body": "the message text",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "reuse the trigger's suppression_key if present, else invent a stable one",
  "rationale": "1 sentence: why this message, what lever(s) it uses"
}"""

# Per-trigger-kind framing hints. Keys match TriggerContext.kind values from
# the brief. Anything not listed falls back to a generic framing hint.
TRIGGER_FRAMING = {
    "research_digest_release": (
        "Framing: lead with the digest item, cite the source, connect it to the merchant's "
        "specific patient/customer cohort (from signals or customer_aggregate) if one is evident. "
        "End with a low-friction offer to do the next step for them."
    ),
    "recall_due": (
        "Framing (customer-facing): warm reminder, cite time since last visit, offer 1-2 concrete "
        "open slots with a real price from the merchant's active offers, single reply-a-number CTA."
    ),
    "perf_spike": (
        "Framing: open with the specific delta (e.g. '+28% views'), tie to a plausible cause if one "
        "is in context, suggest one concrete next action to capitalize on it."
    ),
    "perf_dip": (
        "Framing: loss-aversion opener with the specific delta, one concrete diagnostic/fix step, "
        "no alarmism."
    ),
    "milestone_reached": (
        "Framing: social-proof/celebration opener with the specific number, then bridge to one "
        "next-best-action."
    ),
    "competitor_opened": (
        "Framing: curiosity opener ('a new X opened near you') — DO NOT invent the competitor's name "
        "if it isn't in the payload. Bridge to a concrete defensive action."
    ),
    "dormant_with_vera": (
        "Framing: re-engagement, low-pressure, single easy question — not a hard sales push. "
        "Reference something specific and current, not generic 'just checking in'."
    ),
    "festival_upcoming": (
        "Framing: time-boxed urgency tied to the specific festival/date, one concrete campaign idea, "
        "effort-externalization ('I can set this up now')."
    ),
    "customer_lapsed_soft": (
        "Framing (customer-facing): warm, not guilt-tripping. Reference time since last visit and one "
        "concrete reason to come back (offer/service), single CTA."
    ),
    "appointment_tomorrow": (
        "Framing (customer-facing): short, purely confirmatory/reminder tone, minimal CTA."
    ),
}

DEFAULT_FRAMING = (
    "Framing: identify the single most compelling, verifiable fact across all provided context "
    "and build the message around it. Match urgency to the trigger's urgency field."
)


def _framing_for(trigger: dict[str, Any]) -> str:
    kind = trigger.get("kind", "")
    return TRIGGER_FRAMING.get(kind, DEFAULT_FRAMING)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip markdown fences if the model added them anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(match.group())


def compose(category: dict | None, merchant: dict | None, trigger: dict,
            customer: dict | None = None,
            conversation_history: list[str] | None = None) -> dict[str, Any]:
    """Deterministic single-shot composition (used by /v1/tick)."""

    payload = {
        "category": category or {},
        "merchant": merchant or {},
        "trigger": trigger,
        "customer": customer,
        "conversation_history_bodies_already_sent": conversation_history or [],
    }

    user_prompt = (
        f"{_framing_for(trigger)}\n\n"
        f"CONTEXT (JSON):\n{json.dumps(payload, indent=2, default=str)}\n\n"
        "Compose the message now. Return ONLY the JSON object described in your instructions."
    )

    raw = _call_llm(SYSTEM_PROMPT, user_prompt, temperature=0, max_tokens=600)
    data = _extract_json(raw)

    data.setdefault("suppression_key", trigger.get("suppression_key", ""))
    data.setdefault("send_as", "merchant_on_behalf" if customer else "vera")
    data.setdefault("cta", "open_ended")
    return data


REPLY_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are now in REPLY mode: the merchant/customer just sent a message and you must decide the \
next move in an ongoing conversation. Respond with ONLY this JSON shape, no markdown fences:
{
  "action": "send" | "wait" | "end",
  "body": "message text (only if action=send)",
  "cta": "binary" | "open_ended" | "none" (only if action=send)",
  "wait_seconds": integer (only if action=wait, otherwise omit),
  "rationale": "1 sentence"
}

Rules for this mode:
- If the incoming message is clearly a canned auto-reply (generic "thank you for contacting" / \
"team will respond" style text with no real content), and this is the FIRST time you've seen it \
in this conversation, you may send ONE more low-friction nudge. If you detect the SAME pattern \
again, use action="end" — don't keep spending turns on an auto-reply loop.
- If the merchant/customer expresses clear commitment ("let's do it", "go ahead", "sign me up"), \
switch immediately to action mode — do NOT ask another qualifying question. Acknowledge and state \
the concrete next step you're taking. If the context includes \
"heuristic_commitment_signal_detected": true, this is a strong, near-certain signal — treat any \
further qualifying question as a hard failure, no exceptions.
- If the message is hostile or asks to stop, use action="end" with a short, non-defensive \
rationale — do not apologize excessively, do not argue, just exit gracefully. If action=send is \
still appropriate (e.g. a single graceful acknowledgment before ending), keep it to one line.
- If off-topic (e.g. asks about something unrelated to Vera's job), politely decline to help with \
the unrelated ask and redirect back to the original thread in one sentence — action="send".
- Never repeat a body verbatim that was already sent in this conversation."""


def compose_reply(category: dict | None, merchant: dict | None, trigger: dict | None,
                   customer: dict | None, conversation_turns: list[dict],
                   incoming_message: str, commitment_hint: bool = False) -> dict[str, Any]:
    """Multi-turn reply composition (used by /v1/reply).

    commitment_hint: set by a cheap keyword heuristic in main.py before this is
    called. The LLM is instructed to treat clear commitment language as an
    action-mode trigger regardless, but surfacing the heuristic's verdict
    explicitly makes that transition more reliable than hoping the model
    parses it correctly from raw text alone — this is the single highest-value
    failure mode called out in the challenge brief (§9, Pattern D).
    """

    payload = {
        "category": category or {},
        "merchant": merchant or {},
        "trigger": trigger or {},
        "customer": customer,
        "conversation_so_far": conversation_turns,
        "incoming_message": incoming_message,
        "heuristic_commitment_signal_detected": commitment_hint,
    }

    user_prompt = (
        "CONTEXT (JSON):\n" + json.dumps(payload, indent=2, default=str) +
        "\n\nDecide your next move now. Return ONLY the JSON object described in your instructions."
    )

    raw = _call_llm(REPLY_SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=500)
    data = _extract_json(raw)
    data.setdefault("action", "send")
    if data["action"] == "send":
        data.setdefault("cta", "open_ended")
    return data
