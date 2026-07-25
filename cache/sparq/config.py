from pydantic import BaseModel, ConfigDict, Field

from cache.similarity_cache import RankingDistanceMethod


class SPARQConfig(BaseModel):
    """Configuration for the SPARQ semantic cache policy."""

    model_config = ConfigDict(frozen=True)

    max_size: int = Field(gt=0)
    hit_distance_threshold: float = Field(ge=0.0)
    distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE
    window_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    credit_power: float = Field(default=2.0, gt=0.0)
    aging_interval_requests: int | None = Field(default=None, gt=0)
    aging_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    admission_margin: float = Field(default=0.0, ge=0.0)
    initial_score: float = Field(default=1.0, gt=0.0)

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
    def effective_aging_interval(self) -> int:
        if self.aging_interval_requests is not None:
            return self.aging_interval_requests
        return max(1, 4 * self.max_size)
