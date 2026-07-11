from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cache.similarity_cache import RankingDistanceMethod


class SAGEConfig(BaseModel):
    """Configuration for Semantic Admission and Gain-based Eviction (SAGE)."""

    model_config = ConfigDict(frozen=True)

    max_size: int = Field(gt=0)
    hit_distance_threshold: float = Field(ge=0.0)
    distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE
    ghost_capacity: int = Field(default=4096, gt=0)
    decay_half_life_requests: float | None = Field(default=None, gt=0.0)
    admission_margin: float = Field(default=0.0, ge=0.0)
    storage_path: Path | None = None
    storage_namespace: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("storage_namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if any(character not in allowed for character in value):
            raise ValueError("storage_namespace may contain only letters, digits, '-', '_', and '.'")
        return value
