from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SPARQLookupContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: list[float]


class SPARQDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    admitted: bool
    reason: Literal[
        "concurrent_semantic_duplicate",
        "direct_insert",
        "direct_replace",
        "direct_reject",
        "window_insert",
        "window_replace",
        "main_free_slot",
        "candidate_promoted",
        "candidate_rejected",
    ]
    incoming_key: str
    candidate_key: str | None = None
    victim_key: str | None = None
    victim_slot: int | None = None
    candidate_score: float | None = None
    victim_score: float | None = None
    incoming_admitted: bool = False
    promoted: bool = False


class SPARQStats(BaseModel):
    requests: int = 0
    hits: int = 0
    misses: int = 0
    admissions: int = 0
    rejections: int = 0
    evictions: int = 0
    concurrent_deduplications: int = 0
    window_hits: int = 0
    main_hits: int = 0
    window_insertions: int = 0
    promotions: int = 0
    promotion_rejections: int = 0
    aging_events: int = 0
    lookup_time_ms: float = 0.0
    admission_time_ms: float = 0.0

    @property
    def semantic_hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def average_lookup_ms(self) -> float:
        return self.lookup_time_ms / self.requests if self.requests else 0.0


class SPARQResident(BaseModel):
    slot: int = Field(ge=0)
    segment: Literal["window", "main"]
    key: str
    prompt: str
    response: str
    vector: list[float]
    score: float = Field(ge=0.0)
    inserted_step: int = Field(ge=0)
    last_access_step: int = Field(ge=0)
