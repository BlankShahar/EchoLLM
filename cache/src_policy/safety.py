import re
from typing import Any

_PRIVATE_PATTERNS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b(?:sk|pk|api|token|secret)[-_]?[A-Za-z0-9]{16,}\b", re.I),
    re.compile(r"\b(?:order|account|customer|invoice|ticket)[-_ ]?#?\s*[A-Za-z0-9-]{6,}\b", re.I),
    re.compile(r"\b\+?\d[\d\s().-]{8,}\d\b"),
]

_TIME_MARKERS = (
    "today", "now", "latest", "current", "recent", "this week", "this month",
    "live", "breaking", "right now", "yesterday", "tomorrow",
)

_CREATIVE_MARKERS = (
    "write a poem", "write a story", "brainstorm", "be creative",
    "random", "surprise me", "roleplay",
)

_STABLE_MARKERS = (
    "explain", "define", "what is", "how does", "summarize",
    "teach me", "intuitive explanation", "example of",
)

_HIGH_STAKES_MARKERS = (
    "medical advice", "legal advice", "financial advice", "diagnose",
    "prescription", "lawsuit", "tax advice",
)


def has_private_marker(text: str) -> bool:
    return bool(text) and any(pattern.search(text) for pattern in _PRIVATE_PATTERNS)


def has_time_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _TIME_MARKERS)


def safety_score(prompt: str, response: str, kwargs: dict[str, Any]) -> float:
    lower = prompt.lower()

    if has_private_marker(prompt) or has_private_marker(response):
        return 0.0
    if has_time_marker(prompt):
        return 0.0

    tool_or_rag = kwargs.get("tool_dependent") or kwargs.get("rag_dependent")
    missing_state = "tool_config" not in kwargs and "retrieval_config" not in kwargs
    if tool_or_rag and missing_state:
        return 0.0

    temperature = kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and temperature > 0.7:
        return 0.3
    if any(marker in lower for marker in _CREATIVE_MARKERS):
        return 0.3
    if any(marker in lower for marker in _HIGH_STAKES_MARKERS):
        return 0.4
    if any(marker in lower for marker in _STABLE_MARKERS):
        return 1.0
    return 0.7
