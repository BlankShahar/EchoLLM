from math import ceil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cache.similarity_cache import RankingDistanceMethod


class SAGEConfig(BaseModel):
    """Configuration for Semantic Admission and Gain-based Eviction (SAGE)."""

    model_config = ConfigDict(frozen=True)

    max_size: int = Field(gt=0)
    hit_distance_threshold: float = Field(ge=0.0)
    distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE
    ghost_capacity: int | None = Field(default=None, gt=0)
    decay_half_life_requests: float | None = Field(default=None, gt=0.0)
    admission_margin: float = Field(default=0.0, ge=0.0)
    current_request_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    window_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    soft_coverage: bool = True
    soft_coverage_power: float = Field(default=1.0, gt=0.0)
    recent_history_multiplier: float = Field(default=4.0, gt=0.0)
    recent_history_limit: int = Field(default=4096, gt=0)
    long_history_capacity: int | None = Field(default=None, gt=0)
    long_history_multiplier: float = Field(default=8.0, gt=0.0)
    long_history_limit: int = Field(default=8192, gt=0)
    long_sample_stride: int = Field(default=8, gt=0)
    recent_evidence_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    long_decay_half_life_requests: float | None = Field(default=None, gt=0.0)
    storage_path: Path | None = None
    storage_namespace: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("storage_namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if any(character not in allowed for character in value):
            raise ValueError("storage_namespace may contain only letters, digits, '-', '_', and '.'")
        return value

    @model_validator(mode="after")
    def validate_partition(self) -> "SAGEConfig":
        if self.window_fraction == 1.0 and self.max_size > 1:
            raise ValueError("window_fraction must leave room for the SAGE main cache")
        return self

    @property
    def window_size(self) -> int:
        if self.window_fraction == 0.0:
            return 0
        if self.max_size == 1:
            return 1
        requested = max(1, int(round(self.max_size * self.window_fraction)))
        return min(self.max_size - 1, requested)

    @property
    def main_size(self) -> int:
        return self.max_size - self.window_size

    @property
    def recent_capacity(self) -> int:
        if self.ghost_capacity is not None:
            return self.ghost_capacity
        target = ceil(max(1, self.main_size) * self.recent_history_multiplier)
        return min(self.recent_history_limit, target)

    @property
    def long_capacity(self) -> int:
        if self.long_history_capacity is not None:
            return self.long_history_capacity
        target = ceil(max(1, self.main_size) * self.long_history_multiplier)
        return min(self.long_history_limit, target)
