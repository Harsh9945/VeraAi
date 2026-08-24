"""
Local control-flow tests that mock out the LLM calls in app.composer.
These verify the endpoint logic (idempotency, dedup, auto-reply escalation,
hostility handling, anti-repetition) WITHOUT needing a real ANTHROPIC_API_KEY
or spending any tokens. Run with your real key against a live server (or
judge_simulator.py) separately to verify actual composition quality.

Run: python test_local.py
"""

import os
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.state import store

client = TestClient(app)


def reset():
    store.teardown()


def push(scope, cid, version, payload, expected_status=200):
    r = client.post("/v1/context", json={
        "scope": scope, "context_id": cid, "version": version,
        "payload": payload, "delivered_at": "2026-04-26T10:00:00Z",
    })
    assert r.status_code == expected_status, f"Expected {expected_status}, got {r.status_code}: {r.text}"
    return r.json()


def test_context_idempotency():
    reset()
    r1 = push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists"})
    assert r1["accepted"] is True
    r2 = push("merchant", "m1", 1, {"merchant_id": "m1"}, expected_status=409)
    assert r2["accepted"] is False
    assert r2["current_version"] == 1
    r3 = push("merchant", "m1", 2, {"merchant_id": "m1", "category_slug": "dentists", "v": 2})
    assert r3["accepted"] is True
    print("PASS: context idempotency + version replacement")


def test_healthz_counts():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1"})
    r = client.get("/v1/healthz")
    assert r.json()["contexts_loaded"] == {"category": 1, "merchant": 1, "customer": 0, "trigger": 0}
    print("PASS: healthz reflects loaded contexts")


def test_tick_composes_action():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists",
                                "identity": {"name": "Dr. Meera"}})
    push("trigger", "t1", 1, {"id": "t1", "kind": "research_digest_release",
                               "merchant_id": "m1", "suppression_key": "sk1"})

    fake = {"body": "Dr. Meera, here's a digest item...", "cta": "open_ended",
            "send_as": "vera", "rationale": "test"}
    with patch("app.composer.compose", return_value=fake):
        r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                           "available_triggers": ["t1"]})
    actions = r.json()["actions"]
    assert len(actions) == 1
    assert actions[0]["merchant_id"] == "m1"
    assert actions[0]["body"] == fake["body"]
    print("PASS: tick composes and returns action for a valid trigger")


def test_tick_skips_unknown_trigger_and_missing_merchant():
    reset()
    with patch("app.composer.compose", return_value={"body": "x"}):
        r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                           "available_triggers": ["does_not_exist"]})
    assert r.json()["actions"] == []
    print("PASS: tick returns empty actions for unresolvable triggers")


def test_tick_dedup_on_suppression_key():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists",
                                "identity": {"name": "Dr. Meera"}})
    push("trigger", "t1", 1, {"id": "t1", "kind": "research_digest_release",
                               "merchant_id": "m1", "suppression_key": "sk1"})
    fake = {"body": "msg", "cta": "open_ended", "send_as": "vera", "rationale": "r"}
    with patch("app.composer.compose", return_value=fake):
        r1 = client.post("/v1/tick", json={"now": "t", "available_triggers": ["t1"]})
        r2 = client.post("/v1/tick", json={"now": "t", "available_triggers": ["t1"]})
    assert len(r1.json()["actions"]) == 1
    assert len(r2.json()["actions"]) == 0, "should not re-send same suppression_key"
    print("PASS: tick dedups repeat sends on suppression_key")


def test_hostile_reply_ends_without_llm_call():
    reset()
    with patch("app.composer.compose_reply") as mock_compose:
        r = client.post("/v1/reply", json={
            "conversation_id": "c1", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "Stop messaging me. This is spam.",
            "received_at": "t", "turn_number": 2,
        })
        assert mock_compose.call_count == 0, "hostile message should short-circuit before LLM call"
    assert r.json()["action"] == "end"
    print("PASS: hostile message ends conversation without spending an LLM call")


def test_auto_reply_three_turn_progression():
    reset()
    canned = "Thank you for contacting us! Our team will respond shortly."
    fake_nudge = {"action": "send", "body": "One more nudge", "cta": "open_ended", "rationale": "r"}

    with patch("app.composer.compose_reply", return_value=fake_nudge):
        r1 = client.post("/v1/reply", json={
            "conversation_id": "c2", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": canned, "received_at": "t", "turn_number": 1,
        })
    assert r1.json()["action"] == "send", "first auto-reply-shaped msg allows one more nudge"

    with patch("app.composer.compose_reply", return_value=fake_nudge) as mock_compose:
        r2 = client.post("/v1/reply", json={
            "conversation_id": "c2", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": canned, "received_at": "t", "turn_number": 2,
        })
        assert mock_compose.call_count == 0, "second auto-reply should bypass LLM call"
    assert r2.json()["action"] == "wait"
    assert r2.json()["wait_seconds"] == 86400

    with patch("app.composer.compose_reply", return_value=fake_nudge) as mock_compose:
        r3 = client.post("/v1/reply", json={
            "conversation_id": "c2", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": canned, "received_at": "t", "turn_number": 3,
        })
        assert mock_compose.call_count == 0, "third auto-reply should bypass LLM call"
    assert r3.json()["action"] == "end"
    print("PASS: auto-reply sequence correct: 1st=nudge, 2nd=wait(24h), 3rd=end")



def test_intent_commitment_hint_passed_to_composer():
    reset()
    with patch("app.composer.compose_reply") as mock_compose:
        mock_compose.return_value = {"action": "send", "body": "Done, sending now.",
                                      "cta": "open_ended", "rationale": "r"}
        client.post("/v1/reply", json={
            "conversation_id": "c3", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "Ok lets do it. Whats next?",
            "received_at": "t", "turn_number": 2,
        })
        _, kwargs = mock_compose.call_args
        assert kwargs["commitment_hint"] is True
    print("PASS: commitment language correctly sets the heuristic hint passed to the composer")


def test_no_repeat_body_in_same_conversation():
    reset()
    fake = {"action": "send", "body": "Same message every time", "cta": "open_ended", "rationale": "r"}
    with patch("app.composer.compose_reply", return_value=fake):
        r1 = client.post("/v1/reply", json={
            "conversation_id": "c4", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "ok tell me more", "received_at": "t", "turn_number": 1,
        })
        r2 = client.post("/v1/reply", json={
            "conversation_id": "c4", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "tell me more again", "received_at": "t", "turn_number": 2,
        })
    assert r1.json()["action"] == "send"
    assert r2.json()["action"] == "end", "should refuse to repeat the exact same body verbatim"
    print("PASS: anti-repetition guard ends rather than repeating a body verbatim")


def test_fabrication_check_tick_skips():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists",
                                "identity": {"name": "Dr. Meera"}})
    push("trigger", "t1", 1, {"id": "t1", "kind": "research_digest_release",
                               "merchant_id": "m1", "suppression_key": "sk1"})

    # "500" is not in context
    fake = {"body": "Dr. Meera, you have 500 patients.", "cta": "open_ended",
            "send_as": "vera", "rationale": "test"}
    with patch("app.composer.compose", return_value=fake):
        r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                           "available_triggers": ["t1"]})
    actions = r.json()["actions"]
    assert len(actions) == 0
    print("PASS: fabrication check in tick correctly skips message with invented numbers")


def test_fabrication_check_reply_ends():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists",
                                "identity": {"name": "Dr. Meera"}})
    
    # "999" is not in context
    fake = {"action": "send", "body": "Please pay 999 rupees.", "cta": "open_ended", "rationale": "r"}
    with patch("app.composer.compose_reply", return_value=fake):
        r = client.post("/v1/reply", json={
            "conversation_id": "c_fab", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "how much?", "received_at": "t", "turn_number": 1,
        })
    assert r.json()["action"] == "end"
    assert "Fabrication check failed" in r.json()["rationale"]
    print("PASS: fabrication check in reply correctly ends conversation on invented numbers")


def test_sqlite_persistence_across_instances():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1"})
    
    from app.state import Store
    new_store = Store()
    assert new_store.get_context("category", "dentists") == {"slug": "dentists"}
    assert new_store.get_context("merchant", "m1") == {"merchant_id": "m1"}
    
    conv = new_store.get_or_create_conversation("c_persist", merchant_id="m1")
    from app.state import Turn
    conv.turns.append(Turn(from_role="vera", message="hello persistence", ts="t1"))
    
    new_store_2 = Store()
    conv2 = new_store_2.conversations.get("c_persist")
    assert conv2 is not None
    assert conv2.merchant_id == "m1"
    assert len(conv2.turns) == 1
    assert conv2.turns[0].message == "hello persistence"
    print("PASS: sqlite persistence works across distinct store instances")


def test_url_safety_filters():
    reset()
    push("category", "dentists", 1, {"slug": "dentists"})
    push("merchant", "m1", 1, {"merchant_id": "m1", "category_slug": "dentists",
                                "identity": {"name": "Dr. Meera"}})
    push("trigger", "t1", 1, {"id": "t1", "kind": "research_digest_release",
                               "merchant_id": "m1", "suppression_key": "sk1"})

    # 1. Tick contains a URL -> skipped
    fake = {"body": "Visit http://magicpin.com for more info.", "cta": "open_ended",
            "send_as": "vera", "rationale": "test"}
    with patch("app.composer.compose", return_value=fake):
        r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                           "available_triggers": ["t1"]})
    assert len(r.json()["actions"]) == 0

    # 2. Reply contains a URL -> ended
    fake_reply = {"action": "send", "body": "Go to www.magicpin.in to book.", "cta": "open_ended", "rationale": "r"}
    with patch("app.composer.compose_reply", return_value=fake_reply):
        r = client.post("/v1/reply", json={
            "conversation_id": "c_url", "merchant_id": "m1", "customer_id": None,
            "from_role": "merchant", "message": "give link", "received_at": "t", "turn_number": 1,
        })
    assert r.json()["action"] == "end"
    assert "URL detected" in r.json()["rationale"]
    print("PASS: URL filter correctly skips tick actions and ends replies with links")


if __name__ == "__main__":
    test_context_idempotency()
    test_healthz_counts()
    test_tick_composes_action()
    test_tick_skips_unknown_trigger_and_missing_merchant()
    test_tick_dedup_on_suppression_key()
    test_hostile_reply_ends_without_llm_call()
    test_auto_reply_three_turn_progression()
    test_intent_commitment_hint_passed_to_composer()
    test_no_repeat_body_in_same_conversation()
    test_fabrication_check_tick_skips()
    test_fabrication_check_reply_ends()
    test_sqlite_persistence_across_instances()
    test_url_safety_filters()
    print("\nAll local control-flow tests passed.")

