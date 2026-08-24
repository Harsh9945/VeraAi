"""
Cheap, deterministic checks that run *around* the LLM call.

These exist because the rubric explicitly penalizes things an LLM alone tends
to get wrong under pressure: repeating canned merchant auto-replies back and
forth, missing an obvious "yes let's do it", multiple CTAs, or sending the
exact same body twice in one conversation. None of this needs an LLM call —
a few regex/heuristics catch most of it, and they're cheap enough to run
before every send.
"""

import re
from app.state import ConversationState

# Common WhatsApp Business canned auto-reply phrasing (English + common Hindi
# transliterations). Not exhaustive — the streak check below catches most
# real auto-replies even if the phrase list misses one.
AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "we will get back to you",
    "hamari team tak pahuncha",
    "automated assistant",
    "aap se jaldi hi sampark",
    "currently unavailable",
    "will reply shortly",
    "reaching out to you soon",
]

INTENT_PHRASES = [
    "let's do it", "lets do it", "ok lets do", "ok let's do",
    "go ahead", "sure do it", "yes please proceed", "proceed",
    "haan kar do", "theek hai kar do", "chalo shuru karo", "start karo",
    "sign me up", "i'm in", "im in", "yes i want", "join karna hai", "judrna hai",
]

HOSTILE_PHRASES = [
    "stop messaging", "spam", "leave me alone", "don't message",
    "harass", "fuck off", "get lost", "band karo",
]


def is_probable_auto_reply(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in AUTO_REPLY_PHRASES)


def is_repeated_verbatim(conv: ConversationState, message: str) -> bool:
    """3+ verbatim repeats of the same inbound message = treat as auto-reply,
    per the brief's explicit hint: 'same message verbatim 3+ times = auto-reply.'
    """
    inbound = [t.message.strip() for t in conv.turns if t.from_role in ("merchant", "customer")]
    inbound.append(message.strip())
    return inbound.count(message.strip()) >= 3


def signals_intent_commitment(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in INTENT_PHRASES)


def signals_hostility(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in HOSTILE_PHRASES)


def is_duplicate_body(conv: ConversationState, body: str) -> bool:
    return body.strip() in {b.strip() for b in conv.sent_bodies}


def has_single_cta(cta: str) -> bool:
    # cta is a classified field (binary/open_ended/none) coming out of the
    # composer, not free text — this just guards against an empty/invalid value.
    return cta in ("binary", "open_ended", "none")


def strip_multi_cta_body(body: str) -> str:
    """Best-effort cleanup if the LLM slips in a 'Reply YES for X, NO for Y'
    style multi-branch CTA. Keeps only the first sentence containing 'reply'.
    This is a safety net, not the primary defense (the prompt already
    instructs single-CTA) — most of the time this is a no-op.
    """
    sentences = re.split(r'(?<=[.!?])\s+', body.strip())
    reply_idxs = [i for i, s in enumerate(sentences) if re.search(r'\breply\b', s, re.I)]
    if len(reply_idxs) <= 1:
        return body
    # keep everything up to and including the first "reply" sentence
    return " ".join(sentences[: reply_idxs[0] + 1])


def check_for_fabrications(body: str, contexts: list) -> list[str]:
    """
    Extracts all numbers from `body` and checks if they exist as substrings
    or numeric values in any of the provided context objects.
    Returns a list of numbers that appear to be fabricated (not found in contexts).
    """
    import json
    if not body:
        return []

    # Find all sequences of digits, potentially with decimals
    raw_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', body)

    # Convert all context objects to strings for easy substring matching
    context_strings = []
    for ctx in contexts:
        if ctx:
            if isinstance(ctx, str):
                context_strings.append(ctx)
            else:
                context_strings.append(json.dumps(ctx))

    fabricated = []
    for num in raw_numbers:
        # Ignore single-digit integers (0-9) to avoid false positives on list items, CTA options, etc.
        if re.match(r'^\d$', num):
            continue

        # Check if the number (as a string) exists in any context string
        found = False
        for ctx_str in context_strings:
            if num in ctx_str:
                found = True
                break

        # If not found and it looks like a float ending in .0 (e.g. 28.0), try matching the integer part
        if not found and num.endswith(".0"):
            short_num = num[:-2]
            for ctx_str in context_strings:
                if short_num in ctx_str:
                    found = True
                    break

        if not found:
            fabricated.append(num)

    return fabricated


def contains_url(text: str) -> bool:
    """
    Checks if the given text contains a URL (http://, https://, or www.).
    This helps prevent the -3 per URL penalty.
    """
    if not text:
        return False
    return bool(re.search(r'\b(?:https?://|www\.)\S+\b', text, re.I))


