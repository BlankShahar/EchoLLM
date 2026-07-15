import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .resources import ResourceUsage


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
    capacity_mode: str
    llm_model: str
    measured_requests: int
    hits: int
    misses: int
    hit_rate: float
    mean_hit_response_cosine_distance: float | None
    p95_hit_response_cosine_distance: float | None
    mean_hit_semantic_accuracy: float | None
    p05_hit_semantic_accuracy: float | None
    mean_end_to_end_response_cosine_distance: float
    mean_end_to_end_semantic_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_policy_overhead_ms: float
    policy_throughput_qps: float | None
    sequential_end_to_end_throughput_qps: float | None
    runner_wall_time_seconds: float
    runner_cpu_time_seconds: float
    baseline_process_rss_mb: float | None
    peak_process_rss_mb: float | None
    peak_process_rss_delta_mb: float | None
    runner_throughput_qps: float | None
    quality_adjusted_hit_rates: dict[str, float]
    good_hit_precisions: dict[str, float | None]
    bad_hit_rates: dict[str, float]

    def flat_dict(self) -> dict[str, object]:
        result = self.model_dump(
            exclude={
                "quality_adjusted_hit_rates",
                "good_hit_precisions",
                "bad_hit_rates",
            }
        )
        for threshold, value in self.quality_adjusted_hit_rates.items():
            result[f"quality_adjusted_hit_rate@{threshold}"] = value
        for threshold, value in self.good_hit_precisions.items():
            result[f"good_hit_precision@{threshold}"] = value
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
        self._latencies: list[float] = []
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
        self._latencies.append(observation.total_latency_ms)
        if observation.response_cosine_distance is None:
            raise ValueError("Every delivered response must include response_cosine_distance")
        distance = observation.response_cosine_distance
        self._end_to_end_distance_sum += distance
        if observation.hit:
            self._hits += 1
            self._hit_distances.append(distance)
            for threshold in self._quality_thresholds:
                if distance <= threshold:
                    self._good_hits[threshold] += 1
                else:
                    self._bad_hits[threshold] += 1

    def summary(
        self,
        policy: str,
        cache_size: int,
        *,
        capacity_mode: str = "bounded",
        llm_model: str = "unknown",
        resource_usage: ResourceUsage | None = None,
    ) -> RunSummary:
        if self._requests == 0:
            raise ValueError("No measured requests")
        hit_array = np.asarray(self._hit_distances, dtype=np.float64)
        latency_array = np.asarray(self._latencies, dtype=np.float64)
        mean_hit_distance = float(hit_array.mean()) if hit_array.size else None
        p95_hit_distance = float(np.percentile(hit_array, 95)) if hit_array.size else None
        misses = self._requests - self._hits
        mean_latency = self._latency_sum / self._requests
        mean_end_to_end_distance = self._end_to_end_distance_sum / self._requests
        policy_seconds = self._overhead_sum / 1000.0
        usage = resource_usage or ResourceUsage(
            runner_wall_time_seconds=0.0,
            runner_cpu_time_seconds=0.0,
        )
        return RunSummary(
            policy=policy,
            cache_size=cache_size,
            capacity_mode=capacity_mode,
            llm_model=llm_model,
            measured_requests=self._requests,
            hits=self._hits,
            misses=misses,
            hit_rate=self._hits / self._requests,
            mean_hit_response_cosine_distance=mean_hit_distance,
            p95_hit_response_cosine_distance=p95_hit_distance,
            mean_hit_semantic_accuracy=(
                1.0 - mean_hit_distance if mean_hit_distance is not None else None
            ),
            p05_hit_semantic_accuracy=(
                1.0 - p95_hit_distance if p95_hit_distance is not None else None
            ),
            mean_end_to_end_response_cosine_distance=mean_end_to_end_distance,
            mean_end_to_end_semantic_accuracy=1.0 - mean_end_to_end_distance,
            mean_latency_ms=mean_latency,
            p95_latency_ms=float(np.percentile(latency_array, 95)),
            p99_latency_ms=float(np.percentile(latency_array, 99)),
            mean_policy_overhead_ms=self._overhead_sum / self._requests,
            policy_throughput_qps=(
                self._requests / policy_seconds if policy_seconds > 0.0 else None
            ),
            sequential_end_to_end_throughput_qps=(
                1000.0 / mean_latency if mean_latency > 0.0 else None
            ),
            runner_wall_time_seconds=usage.runner_wall_time_seconds,
            runner_cpu_time_seconds=usage.runner_cpu_time_seconds,
            baseline_process_rss_mb=usage.baseline_process_rss_mb,
            peak_process_rss_mb=usage.peak_process_rss_mb,
            peak_process_rss_delta_mb=usage.peak_process_rss_delta_mb,
            runner_throughput_qps=usage.runner_throughput_qps,
            quality_adjusted_hit_rates={
                _threshold_key(threshold): count / self._requests
                for threshold, count in self._good_hits.items()
            },
            good_hit_precisions={
                _threshold_key(threshold): (
                    count / self._hits if self._hits else None
                )
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
