import math

from _experiments.metrics import MetricsAccumulator, RequestObservation


def test_quality_adjusted_hit_rate_uses_all_requests_as_denominator() -> None:
    metrics = MetricsAccumulator([0.2])
    metrics.record(
        RequestObservation(
            measured=True,
            hit=True,
            response_cosine_distance=0.1,
            total_latency_ms=1.0,
            policy_overhead_ms=1.0,
        )
    )
    metrics.record(
        RequestObservation(
            measured=True,
            hit=True,
            response_cosine_distance=0.4,
            total_latency_ms=1.0,
            policy_overhead_ms=1.0,
        )
    )
    metrics.record(
        RequestObservation(
            measured=True,
            hit=False,
            response_cosine_distance=0.2,
            total_latency_ms=10.0,
            policy_overhead_ms=1.0,
        )
    )
    summary = metrics.summary("SAGE", 10)
    assert summary.hit_rate == 2 / 3
    assert summary.quality_adjusted_hit_rates["0.2"] == 1 / 3
    assert summary.good_hit_precisions["0.2"] == 1 / 2
    assert summary.bad_hit_rates["0.2"] == 1 / 3
    assert math.isclose(summary.mean_hit_semantic_accuracy or 0.0, 0.75)
    assert math.isclose(summary.mean_end_to_end_semantic_accuracy, 1.0 - (0.7 / 3.0))
    assert math.isclose(summary.p95_latency_ms, 9.1)
    assert math.isclose(summary.p99_latency_ms, 9.82)
    assert math.isclose(summary.policy_throughput_qps or 0.0, 1000.0)
    assert math.isclose(summary.sequential_end_to_end_throughput_qps or 0.0, 250.0)
