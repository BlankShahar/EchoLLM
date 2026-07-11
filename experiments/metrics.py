import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RequestObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    measured: bool
    hit: bool
    response_cosine_distance: float | None = None
    total_latency_ms: float = Field(ge=0.0)
    policy_overhead_ms: float = Field(ge=0.0)


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy: str
    cache_size: int
    measured_requests: int
    hits: int
    misses: int
    hit_rate: float
    mean_hit_response_cosine_distance: float | None
    p95_hit_response_cosine_distance: float | None
    mean_end_to_end_response_cosine_distance: float
    mean_latency_ms: float
    mean_policy_overhead_ms: float
    quality_adjusted_hit_rates: dict[str, float]
    bad_hit_rates: dict[str, float]

    def flat_dict(self) -> dict[str, object]:
        result = self.model_dump(exclude={"quality_adjusted_hit_rates", "bad_hit_rates"})
        for threshold, value in self.quality_adjusted_hit_rates.items():
            result[f"quality_adjusted_hit_rate@{threshold}"] = value
        for threshold, value in self.bad_hit_rates.items():
            result[f"bad_hit_rate@{threshold}"] = value
        return result


class MetricsAccumulator:
    def __init__(self, quality_thresholds: Sequence[float]):
        self._quality_thresholds = tuple(float(value) for value in quality_thresholds)
        self._requests = 0
        self._hits = 0
        self._latency_sum = 0.0
        self._overhead_sum = 0.0
        self._hit_distances: list[float] = []
        self._end_to_end_distance_sum = 0.0
        self._good_hits = {threshold: 0 for threshold in self._quality_thresholds}
        self._bad_hits = {threshold: 0 for threshold in self._quality_thresholds}

    def record(self, observation: RequestObservation) -> None:
        if not observation.measured:
            return
        self._requests += 1
        self._latency_sum += observation.total_latency_ms
        self._overhead_sum += observation.policy_overhead_ms
        if observation.hit:
            if observation.response_cosine_distance is None:
                raise ValueError("A cache hit must include response_cosine_distance")
            distance = observation.response_cosine_distance
            self._hits += 1
            self._hit_distances.append(distance)
            self._end_to_end_distance_sum += distance
            for threshold in self._quality_thresholds:
                if distance <= threshold:
                    self._good_hits[threshold] += 1
                else:
                    self._bad_hits[threshold] += 1

    def summary(self, policy: str, cache_size: int) -> RunSummary:
        if self._requests == 0:
            raise ValueError("No measured requests")
        hit_array = np.asarray(self._hit_distances, dtype=np.float64)
        mean_hit_distance = float(hit_array.mean()) if hit_array.size else None
        p95_hit_distance = float(np.percentile(hit_array, 95)) if hit_array.size else None
        misses = self._requests - self._hits
        return RunSummary(
            policy=policy,
            cache_size=cache_size,
            measured_requests=self._requests,
            hits=self._hits,
            misses=misses,
            hit_rate=self._hits / self._requests,
            mean_hit_response_cosine_distance=mean_hit_distance,
            p95_hit_response_cosine_distance=p95_hit_distance,
            mean_end_to_end_response_cosine_distance=(
                self._end_to_end_distance_sum / self._requests
            ),
            mean_latency_ms=self._latency_sum / self._requests,
            mean_policy_overhead_ms=self._overhead_sum / self._requests,
            quality_adjusted_hit_rates={
                _threshold_key(threshold): count / self._requests
                for threshold, count in self._good_hits.items()
            },
            bad_hit_rates={
                _threshold_key(threshold): count / self._requests
                for threshold, count in self._bad_hits.items()
            },
        )


def _threshold_key(threshold: float) -> str:
    if math.isclose(threshold, round(threshold)):
        return str(int(round(threshold)))
    return f"{threshold:g}"
