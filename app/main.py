"""
magicpin AI Challenge — bot entrypoint.

Implements the 5 endpoints from testing-brief.md §2:
  POST /v1/context   — receive context pushes (idempotent per (id, version))
  POST /v1/tick      — periodic wake-up, bot may proactively send
  POST /v1/reply     — receive merchant/customer reply, must respond sync
  GET  /v1/healthz   — liveness
  GET  /v1/metadata  — team/bot identity

Run: uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from app import composer, heuristics
from app.state import Turn, store

app = FastAPI(title="Vera Challenge Bot")
START_TIME = time.time()

MAX_ACTIONS_PER_TICK = 20  # per rate-limit table in testing-brief.md §5
MAX_UNANSWERED_NUDGES = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
@app.head("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": store.counts(),
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Team Harsh"),
        "team_members": [m.strip() for m in os.environ.get("TEAM_MEMBERS", "Harsh").split(",")],
        "model": composer.MODEL,
        "approach": "per-trigger-kind prompt routing + deterministic compose, "
                    "heuristic auto-reply/intent/hostility pre-checks around a "
                    "conversational reply composer, in-memory versioned context store",
        "contact_email": os.environ.get("CONTACT_EMAIL", "you@example.com"),
        "version": "1.0.0",
        "submitted_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    accepted, current_version = store.push_context(body.scope, body.context_id, body.version, body.payload)
    if not accepted:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": current_version}
        )
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _resolve_bundle(trigger: dict) -> tuple[dict | None, dict | None, dict | None]:
    """Given a trigger payload, resolve merchant/category/customer contexts."""
    merchant_id = trigger.get("merchant_id")
    customer_id = trigger.get("customer_id")

    merchant = store.get_context("merchant", merchant_id) if merchant_id else None
    category_slug = merchant.get("category_slug") if merchant else None
    category = store.get_context("category", category_slug) if category_slug else None
    customer = store.get_context("customer", customer_id) if customer_id else None

    return merchant, category, customer


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: list[dict[str, Any]] = []

    for trigger_id in body.available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        trigger = store.get_context("trigger", trigger_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        merchant, category, customer = _resolve_bundle(trigger)
        if merchant is None:
            # Can't compose without at least a merchant — skip rather than hallucinate.
            continue

        suppression_key = trigger.get("suppression_key", f"trg:{trigger_id}")

        # Dedup: don't re-send if we've already sent for this suppression_key
        # in an existing (non-ended) conversation tied to this merchant.
        already_sent = any(
            conv.merchant_id == merchant_id and suppression_key in conv.sent_bodies
            for conv in store.conversations.values()
        )
        if already_sent:
            continue

        try:
            import time
            time.sleep(4.0)
            composed = composer.compose(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
                conversation_history=[],
            )
        except Exception as e:  # noqa: BLE001 — never let a bad LLM call kill the tick
            import traceback
            traceback.print_exc()
            continue

        body_text = composed.get("body", "")
        if not body_text:
            continue

        # URL check: prevent sending links that cause a -3 penalty
        if heuristics.contains_url(body_text):
            print(f"WARNING: URL detected in tick: '{body_text}'. Skipping action.")
            continue

        # Fabrication check: ensure no invented numbers are sent
        fab_nums = heuristics.check_for_fabrications(body_text, [category, merchant, trigger, customer])
        if fab_nums:
            print(f"WARNING: Fabrication detected in tick: {fab_nums} in '{body_text}'. Skipping action.")
            continue

        conversation_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}"
        conv = store.get_or_create_conversation(
            conversation_id, merchant_id=merchant_id,
            customer_id=trigger.get("customer_id"), trigger_id=trigger_id,
        )
        conv.turns.append(Turn(from_role="vera", message=body_text, ts=_now_iso()))
        conv.sent_bodies.append(body_text)
        conv.sent_bodies.append(suppression_key)  # also track key for dedup check above

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": trigger.get("customer_id"),
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = store.get_or_create_conversation(
        body.conversation_id, merchant_id=body.merchant_id, customer_id=body.customer_id,
    )

    # --- heuristic pre-checks, before spending an LLM call ---

    if heuristics.signals_hostility(body.message):
        conv.turns.append(Turn(from_role=body.from_role, message=body.message, ts=_now_iso()))
        conv.ended = True
        return {"action": "end", "rationale": "Merchant signaled hostility/opt-out; exiting gracefully."}

    is_auto = heuristics.is_probable_auto_reply(body.message) or heuristics.is_repeated_verbatim(conv, body.message)
    conv.turns.append(Turn(from_role=body.from_role, message=body.message, ts=_now_iso()))

    if is_auto:
        conv.unanswered_nudges += 1
        if conv.unanswered_nudges == 2:
            return {"action": "wait", "wait_seconds": 86400, "rationale": "Same auto-reply twice in a row → owner not at phone. Wait 24h before retry."}
        elif conv.unanswered_nudges >= 3:
            conv.ended = True
            return {"action": "end", "rationale": "Auto-reply 3x in a row, no real reply. Closing."}
        # first time seeing an auto-reply-shaped message: allow one composed nudge below
    else:
        conv.unanswered_nudges = 0

    # --- resolve context for this conversation ---

    trigger = store.get_context("trigger", conv.trigger_id) if conv.trigger_id else {}
    merchant, category, customer = (None, None, None)
    if body.merchant_id:
        merchant = store.get_context("merchant", body.merchant_id)
        if merchant:
            category = store.get_context("category", merchant.get("category_slug"))
    if body.customer_id:
        customer = store.get_context("customer", body.customer_id)

    conversation_turns = [{"from": t.from_role, "message": t.message} for t in conv.turns]
    commitment_hint = heuristics.signals_intent_commitment(body.message)

    try:
        result = composer.compose_reply(
            category=category, merchant=merchant, trigger=trigger, customer=customer,
            conversation_turns=conversation_turns, incoming_message=body.message,
            commitment_hint=commitment_hint,
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"action": "wait", "wait_seconds": 300, "rationale": f"Internal error composing reply: {e}"}

    action = result.get("action", "send")

    if action == "send":
        out_body = result.get("body", "")
        if not out_body or heuristics.is_duplicate_body(conv, out_body):
            return {"action": "end", "rationale": "Nothing new to say without repeating a prior message."}

        # URL check: prevent sending links that cause a -3 penalty
        if heuristics.contains_url(out_body):
            print(f"WARNING: URL detected in reply: '{out_body}'. Ending conversation.")
            conv.ended = True
            return {"action": "end", "rationale": "URL detected in generated message."}

        # Fabrication check: ensure no invented numbers are sent
        fab_nums = heuristics.check_for_fabrications(out_body, [category, merchant, trigger, customer, conversation_turns])
        if fab_nums:
            print(f"WARNING: Fabrication detected in reply: {fab_nums} in '{out_body}'. Ending conversation.")
            conv.ended = True
            return {"action": "end", "rationale": f"Fabrication check failed. Invented numbers: {fab_nums}."}
        out_body = heuristics.strip_multi_cta_body(out_body)
        conv.turns.append(Turn(from_role="vera", message=out_body, ts=_now_iso()))
        conv.sent_bodies.append(out_body)
        return {
            "action": "send",
            "body": out_body,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", ""),
        }

    if action == "wait":
        return {
            "action": "wait",
            "wait_seconds": int(result.get("wait_seconds", 1800)),
            "rationale": result.get("rationale", ""),
        }

    conv.ended = True
    return {"action": "end", "rationale": result.get("rationale", "Ending conversation.")}


# ---------------------------------------------------------------------------
# POST /v1/teardown  (optional, per testing-brief.md §11)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    store.teardown()
    return {"status": "wiped"}
