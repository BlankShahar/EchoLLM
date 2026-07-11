from experiments.metrics import MetricsAccumulator, RequestObservation


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
            total_latency_ms=10.0,
            policy_overhead_ms=1.0,
        )
    )
    summary = metrics.summary("SAGE", 10)
    assert summary.hit_rate == 2 / 3
    assert summary.quality_adjusted_hit_rates["0.2"] == 1 / 3
    assert summary.bad_hit_rates["0.2"] == 1 / 3
