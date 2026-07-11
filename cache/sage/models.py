from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SAGELookupContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: list[float]
    observation_id: int
    request_step: int


class SAGEDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    admitted: bool
    reason: Literal[
        "free_slot",
        "positive_replacement_gain",
        "non_positive_gain",
        "concurrent_semantic_duplicate",
    ]
    candidate_key: str
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


class SAGEStorageMetadata(BaseModel):
    distance_method: str
    hit_distance_threshold: float
    vector_dimension: int = Field(gt=0)
