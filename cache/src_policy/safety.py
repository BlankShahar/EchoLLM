"""
safety.py — Deterministic safety and reusability scoring for SRC.

All checks are cheap: lowercase phrase matching and compiled regular
expressions.  No LLM classifier is used inside the cache policy.

Score table (from the spec)
---------------------------
Private pattern detected              → 0.0
Time-sensitive phrase detected        → 0.0
Tool / RAG-dependent answer detected  → 0.0
Creative / random / high-temperature  → 0.3
Stable educational or factual prompt  → 1.0
Otherwise                             → 0.7
"""

from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Compiled patterns  (module-level so they are compiled once)
# ---------------------------------------------------------------------------

# --- Private / sensitive data ---
_PRIVATE_PATTERNS: list[re.Pattern[str]] = [
    # e-mail addresses
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    # API keys / tokens: long hex / base64-ish strings (≥ 20 chars, mixed case/digits)
    re.compile(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b"),
    # Typical secret-looking tokens (sk-..., Bearer ..., token=...)
    re.compile(r"\b(?:sk|pk|rk|Bearer|token|api[-_]?key)\s*[=:]\s*\S{8,}", re.IGNORECASE),
    # Phone numbers (E.164 or common US/international formats)
    re.compile(r"\+?1?\s*[-.]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"),
    # Account / order / invoice / customer / ticket IDs
    re.compile(
        r"\b(?:account|order|invoice|customer|ticket|user[-_]?id|txn|transaction)[\s:#_-]*\d{4,}\b",
        re.IGNORECASE,
    ),
    # Long secret-looking alphanumeric strings (≥ 32 chars, no common word chars)
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),
]

# --- Time-sensitive phrases (exact substring matching after lower-casing) ---
_TIME_SENSITIVE_PHRASES: frozenset[str] = frozenset(
    [
        "today", "right now", "now", "latest", "current", "recent",
        "this week", "this month", "this year", "breaking", "live",
        "at this moment", "as of today", "as of now", "just released",
        "just announced", "yesterday", "last night", "tonight",
        "this morning", "this afternoon", "this evening",
        "real-time", "realtime", "up-to-date", "up to date",
        "currently", "presently", "nowadays",
    ]
)

# --- Stable educational / factual phrases ---
_EDUCATIONAL_PHRASES: frozenset[str] = frozenset(
    [
        "explain", "define", "what is", "what are", "how does", "how do",
        "summarize", "summarise", "teach me", "example of", "examples of",
        "describe", "difference between", "compare", "history of",
        "overview of", "introduction to", "what does", "why does",
        "how is", "what was", "who is", "who was", "when did",
        "give me an example", "show me how", "walk me through",
        "step by step", "tutorial", "guide to", "how to",
    ]
)

# --- Creative / random / high-temperature generation phrases ---
_CREATIVE_PHRASES: frozenset[str] = frozenset(
    [
        "write a story", "write me a story", "tell me a story",
        "write a poem", "write me a poem", "compose a poem",
        "write a joke", "tell me a joke", "make up",
        "random", "surprise me", "invent", "imagine",
        "creative", "fiction", "fictional", "roleplay", "role-play",
        "generate a name", "generate names", "brainstorm",
        "write a song", "compose a song", "write lyrics",
    ]
)

# --- Tool / RAG-dependent answer markers (in the *response*) ---
_RAG_MARKERS: frozenset[str] = frozenset(
    [
        "according to the document", "based on the document",
        "as stated in the file", "from the attached", "from the uploaded",
        "retrieved from", "search results show", "web search",
        "based on the provided context", "context provided",
        "in the context above", "using the tool",
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_private_pattern(text: str) -> bool:
    """Return True if *text* appears to contain private / sensitive data."""
    for pattern in _PRIVATE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_time_sensitive_phrase(text: str) -> bool:
    """Return True if *text* contains a time-sensitive phrase."""
    lower = text.lower()
    return any(phrase in lower for phrase in _TIME_SENSITIVE_PHRASES)


def _has_educational_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _EDUCATIONAL_PHRASES)


def _has_creative_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _CREATIVE_PHRASES)


def _has_rag_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _RAG_MARKERS)


@lru_cache(maxsize=4096)
def safety_score(prompt: str, response: str) -> float:
    """
    Compute the safety-and-reusability score R ∈ {0.0, 0.3, 0.7, 1.0}.

    The result is cached so repeated calls with identical (prompt, response)
    pairs are free.

    Parameters
    ----------
    prompt : str
        The user prompt (already normalised / stripped).
    response : str
        The LLM-generated response to be cached.

    Returns
    -------
    float
        Safety score:
        - 0.0  → must NOT cache (private data, time-sensitive, tool-dependent)
        - 0.3  → may cache with caution (creative / random)
        - 0.7  → probably fine (default)
        - 1.0  → stable educational / factual — ideal to cache
    """
    # Hard rejections — order matters: privacy first
    if has_private_pattern(prompt) or has_private_pattern(response):
        return 0.0

    if has_time_sensitive_phrase(prompt):
        return 0.0

    if _has_rag_marker(response):
        return 0.0

    # Soft downgrade — creative / high-temperature
    if _has_creative_phrase(prompt):
        return 0.3

    # Ideal for caching — stable educational / factual
    if _has_educational_phrase(prompt):
        return 1.0

    # Default fallback
    return 0.7


def is_safe_to_cache(prompt: str, response: str, theta_safe: float = 0.5) -> bool:
    """Convenience wrapper: True iff safety_score ≥ theta_safe."""
    return safety_score(prompt, response) >= theta_safe


def is_safe_to_reuse(prompt: str, cached_safety: float, theta_safe: float = 0.5) -> bool:
    """
    Check whether a *new incoming* prompt can safely reuse a cached response.

    A cached item is reusable only if:
    - The *incoming* prompt contains no private pattern.
    - The *incoming* prompt contains no time-sensitive phrase.
    - The cached item's own safety score (stored at admission) is ≥ theta_safe.
    """
    if has_private_pattern(prompt):
        return False
    if has_time_sensitive_phrase(prompt):
        return False
    if cached_safety < theta_safe:
        return False
    return True