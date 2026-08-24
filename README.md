# Vera Challenge Bot

A submission for the magicpin AI Challenge. Implements the 5 endpoints from
`testing-brief.md` on top of a composer that targets the 5-dimension rubric
from `challenge-brief.md` directly.

## Architecture (30-second version)

```
app/
├── main.py        # FastAPI app — the 5 required endpoints + optional teardown
├── composer.py     # LLM prompt templates + per-trigger-kind framing, compose() and compose_reply()
├── heuristics.py   # Cheap deterministic checks: auto-reply detection, hostility,
│                   # intent-commitment keyword hint, anti-repetition, multi-CTA cleanup
└── state.py        # In-memory versioned context store + per-conversation state
```

**Why this shape:**

- **One composer, two modes.** `compose()` is deterministic (temp=0) for `/v1/tick` —
  same inputs must produce the same output. `compose_reply()` runs warmer (temp=0.3)
  for `/v1/reply` since genuine conversation needs some variance, but every reply
  still goes through the same rubric-encoding system prompt.
- **Heuristics run *before* the LLM, not instead of it.** Hostility detection and
  repeat-auto-reply detection short-circuit straight to `action: end` without
  spending an LLM call — this is both cheaper and more reliable than trusting the
  model to catch it every time under a 30s budget. The intent-commitment check
  is different: it doesn't replace the LLM call, it *hints* the LLM (see
  `challenge-brief.md` §9 Pattern D — this is called out as production Vera's
  most common failure, so I wanted a belt-and-suspenders fix, not just a prompt
  instruction).
- **Per-trigger-kind framing.** `research_digest_release` and `recall_due` and
  `perf_dip` all want structurally different messages even though they share the
  same 4-context shape — a `TRIGGER_FRAMING` dict injects the right shape hint
  rather than relying on one giant prompt to intuit it every time.
- **Dedup on `suppression_key`.** `/v1/tick` won't re-send to the same merchant
  for the same `suppression_key` if a conversation already exists for it.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set your real ANTHROPIC_API_KEY
```

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Test without spending API calls

```bash
python test_local.py
```

Mocks the LLM layer and verifies the control-flow logic: context idempotency,
tick dedup, hostility short-circuit, auto-reply escalation, intent-hint passing,
anti-repetition. All 9 cases should pass in a couple seconds.

## Test composition quality (needs a real key + costs tokens)

Use magicpin's `judge_simulator.py` against your running server:

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Edit the `LLM_PROVIDER` / `LLM_API_KEY` at the top of `judge_simulator.py` first
(that's the *judge's* LLM, separate from your bot's `ANTHROPIC_API_KEY` in `.env`).

## Deploying for submission

You need a **public HTTPS URL**. Quickest paths:

**Option A — Railway / Render (recommended, minimal config)**
1. Push this folder to a GitHub repo.
2. Railway: "New Project" → "Deploy from GitHub" → set `ANTHROPIC_API_KEY` as
   an environment variable → it auto-detects `requirements.txt` and runs
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (set this as the start
   command explicitly if it doesn't auto-detect).
3. Render: "New Web Service" → same idea, start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

**Option B — ngrok (fastest for testing, not for final submission)**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 &
ngrok http 8080
# submit the https://xxxx.ngrok-free.app URL
```
Note: free ngrok URLs rotate on restart and can be flaky for a 60-minute test
window — fine for local iteration, risky for the actual scored run.

**Option C — Fly.io**
```bash
fly launch   # accept defaults, it'll detect the Dockerfile-less Python app
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

Whichever you pick, hit `GET https://<your-url>/v1/healthz` from a browser or
curl on another network before submitting — confirms it's actually public.

## What's stubbed / what you should extend before submitting

- **Retrieval over `digest` items**: right now the whole category context (all
  digest items) goes into the prompt every time. For the 5-category, small
  dataset here that's fine and even simpler/more reliable than retrieval. If
  digest lists grow large, add embedding-based retrieval to pick the most
  relevant 1-2 items per trigger (mentioned as an option in
  `challenge-brief.md` §13).
- **Fabrication check**: the prompt instructs strongly against inventing data,
  and temp=0 helps, but there's no automated post-hoc check that every number
  in the output actually appears in the input context. Given the -2 penalty
  for fabrication in the rubric, a regex-based "every number in body must
  appear somewhere in the context JSON" check would be a cheap, high-value
  addition if you have time.
- **Multi-turn cadence planning** (brief §12, open challenge #3): currently
  each conversation is independent; there's no cross-conversation planner
  deciding the *sequence* of nudges across a 24h window. Tiebreaker only, per
  the brief — skip unless everything else is solid first.
- **Persistence**: in-memory only, as the brief allows. If your host restarts
  mid-test you lose state — pick a host with no idle-restart behavior for the
  actual scored window, or add a Redis/SQLite layer if time allows.

## Pre-flight checklist (from testing-brief.md §12)

- [ ] Public HTTPS URL reachable from outside your network
- [ ] `python test_local.py` passes
- [ ] `judge_simulator.py` gives non-zero scores against your deployed URL
- [ ] `/v1/tick` and `/v1/reply` both return well within 30s (they should —
      one LLM call each, no chained calls)
- [ ] `ANTHROPIC_API_KEY` has enough quota for a 60-minute test window
- [ ] Submitted URL via the challenge portal
