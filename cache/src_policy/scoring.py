import math

from .models import SRCConfig, SRCMeta


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else max(-1.0, min(1.0, dot / (na * nb)))


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def saved_cost(latency_ms: float, total_tokens: int) -> float:
    return max(0.0, latency_ms) * math.log1p(max(0, total_tokens))


def quality(meta: SRCMeta, config: SRCConfig) -> float:
    if meta.semantic_hits == 0:
        return 1.0
    return max(config.q_min, min(1.0, meta.avg_semantic_similarity))


def freshness(meta: SRCMeta, now: float, config: SRCConfig) -> float:
    return math.exp(-config.freshness_decay * max(0.0, now - meta.created_at))


def eviction_value(meta: SRCMeta, now: float, config: SRCConfig) -> float:
    demand = 1.0 + (
            meta.exact_hits + meta.semantic_hits + 0.5 * meta.near_misses
    ) * math.exp(-config.demand_decay * max(0.0, now - meta.last_access_at))

    return (
            demand * meta.saved_cost * quality(meta, config) * freshness(meta, now, config) * meta.safety_score
    ) / max(1.0, meta.storage_size)
