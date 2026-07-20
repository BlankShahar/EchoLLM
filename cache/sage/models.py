from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SAGELookupContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: list[float]
    observation_id: int
    request_step: int


class SAGEDecision(BaseModel):
    """Admission decision.

    During a window turnover, ``candidate_key`` is the resident leaving probation
    and ``admitted`` says whether it entered the main cache. The request that
    triggered the turnover is identified by ``incoming_key`` and is always placed
    in the probation window when ``incoming_admitted`` is true.
    """

    model_config = ConfigDict(frozen=True)

    admitted: bool
    reason: Literal[
        "free_slot",
        "positive_replacement_gain",
        "non_positive_gain",
        "concurrent_semantic_duplicate",
        "window_insert",
        "window_candidate_promoted",
        "window_candidate_rejected",
        "main_free_slot",
    ]
    candidate_key: str
    incoming_key: str | None = None
    incoming_admitted: bool | None = None
    promoted: bool | None = None
    victim_key: str | None = None
    victim_slot: int | None = None
    candidate_new_gain: float = 0.0
    victim_unrecovered_loss: float = 0.0
    net_delta: float = 0.0


class SAGEStats(BaseModel):
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
    lookup_time_ms: float = 0.0
    admission_time_ms: float = 0.0

    @property
    def semantic_hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def average_lookup_ms(self) -> float:
        return self.lookup_time_ms / self.requests if self.requests else 0.0

    @property
    def average_admission_ms(self) -> float:
        decisions = self.admissions + self.rejections
        return self.admission_time_ms / decisions if decisions else 0.0


class PersistedResident(BaseModel):
    slot: int = Field(ge=0)
    key: str
    prompt: str
    response: str
    vector: list[float]
    inserted_step: int = Field(ge=0)
    last_access_step: int = Field(ge=0)
    frequency: int = Field(default=1, ge=1)


class SAGEStorageMetadata(BaseModel):
    distance_method: str
    hit_distance_threshold: float
    vector_dimension: int = Field(gt=0)
    window_size: int = Field(default=0, ge=0)
    soft_coverage: bool = False
    soft_coverage_power: float = Field(default=1.0, gt=0.0)
    frequency_weight: float = Field(default=0.0, ge=0.0)
