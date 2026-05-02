from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass(slots=True)
class SRCConfig:
    k: int = 8
    sample_size: int = 16
    theta_hit: float = 0.90
    theta_near: float = 0.80
    theta_safe: float = 0.50
    admission_margin: float = 0.05
    demand_decay: float = 1e-4
    ghost_decay: float = 1e-4
    freshness_decay: float = 1e-6
    ghost_weight: float = 0.50
    ghost_max_size: Optional[int] = None
    ghost_sample_size: int = 128
    q_min: float = 0.10
    seed: int = 13
    latency_base_ms: float = 200.0
    latency_per_token_ms: float = 2.0
    max_prompt_chars_for_embedding: int = 8192


@dataclass(slots=True)
class SRCMeta:
    key: str
    prompt: str
    namespace: str
    embedding: list[float]
    created_at: float
    last_access_at: float
    saved_cost: float
    storage_size: float
    safety_score: float
    total_tokens: int
    exact_hits: int = 0
    semantic_hits: int = 0
    near_misses: int = 0
    avg_semantic_similarity: float = 0.0
    ttl_seconds: Optional[float] = None
    invalid: bool = False


@dataclass(slots=True)
class SRCGhost:
    embedding: list[float]
    namespace: str
    timestamp: float
    estimated_cost: float
    reason: str


@dataclass(slots=True)
class Lookup:
    prompt: str
    namespace: str
    normalized: str
    embedding: list[float] = field(default_factory=list)
    hit: bool = False
    hit_type: Literal['exact', 'semantic', 'miss'] = 'miss'
    key: Optional[str] = None
    similarity: float = 0.0
    near_keys: list[str] = field(default_factory=list)
