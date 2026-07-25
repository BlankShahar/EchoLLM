import json
from pathlib import Path

import pandas as pd

from _experiments.config import ExperimentConfig, PolicyConfig
from _experiments.merge_results import (
    merge_policy_results,
    validate_incremental_setup,
)


BASELINE_POLICIES = ["LRU", "LFU", "FIFO", "RR", "SAGE"]


def test_select_grid_builds_a_positive_sparq_only_replay() -> None:
    config = ExperimentConfig(
        policy=PolicyConfig(
            policies=BASELINE_POLICIES,
            cache_sizes=[0, 1_000, 2_000],
            include_unbounded_cache=True,
        )
    )

    selected = config.select_grid(
        policies=["SPARQ"],
        positive_cache_sizes_only=True,
    )

    assert selected.policy.policies == ["SPARQ"]
    assert selected.policy.cache_sizes == [1_000, 2_000]
    assert selected.policy.include_unbounded_cache is False


def test_merge_adds_sparq_and_policy_independent_endpoints(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    incremental = tmp_path / "incremental"
    output = tmp_path / "comparison"
    _write_result_directories(baseline, incremental)

    merged = merge_policy_results(baseline, incremental, output)

    summary = pd.read_csv(merged / "summary.csv")
    bounded = summary.loc[summary["cache_size"] == 1_000]
    zero = summary.loc[summary["cache_size"] == 0]
    assert set(bounded["policy"]) == set(BASELINE_POLICIES) | {"SPARQ"}
    assert set(zero["policy"]) == {"LRU", "SPARQ"}
    assert (merged / "merge_manifest.json").is_file()
    assert (merged / "plots" / "hit_rate_vs_cache_size.png").is_file()
    assert (merged / "plots" / "semantic_accuracy_vs_cache_size.png").is_file()
    assert (baseline / "raw" / "lru_cache_0.csv.gz").is_file()
    assert (incremental / "raw" / "sparq_cache_1000.csv.gz").is_file()


def test_merge_rejects_a_different_raw_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline = tmp_path / "baseline"
    incremental = tmp_path / "incremental"
    _write_result_directories(baseline, incremental)
    _write_raw(
        incremental / "raw" / "sparq_cache_1000.csv.gz",
        response_id="different-response",
    )
    monkeypatch.setattr("_experiments.merge_results.generate_plots", lambda _: [])

    try:
        merge_policy_results(
            baseline,
            incremental,
            tmp_path / "comparison",
        )
    except RuntimeError as error:
        assert "different request traces" in str(error)
    else:
        raise AssertionError("Expected a trace mismatch to reject the merge")


def test_preflight_rejects_missing_baseline_capacity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    incremental = tmp_path / "incremental"
    _write_result_directories(baseline, incremental)
    source = tmp_path / "source.yaml"
    source.write_text(
        ExperimentConfig(
            policy=PolicyConfig(
                policies=BASELINE_POLICIES + ["SPARQ"],
                cache_sizes=[0, 1_000, 2_000],
                include_unbounded_cache=False,
            )
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    try:
        validate_incremental_setup(baseline, source)
    except RuntimeError as error:
        assert "do not match the incremental capacities" in str(error)
    else:
        raise AssertionError("Expected a capacity mismatch to fail preflight")


def _write_result_directories(
    baseline: Path,
    incremental: Path,
) -> None:
    baseline.mkdir()
    incremental.mkdir()
    baseline_rows = [
        _summary_row(policy, 1_000, "bounded")
        for policy in BASELINE_POLICIES
    ]
    baseline_rows.append(_summary_row("LRU", 0, "no_cache"))
    pd.DataFrame(baseline_rows).to_csv(baseline / "summary.csv", index=False)
    pd.DataFrame(
        [_summary_row("SPARQ", 1_000, "bounded")]
    ).to_csv(incremental / "summary.csv", index=False)

    stats = {"trace_requests": 1, "unique_prompt_strings": 1}
    for directory in (baseline, incremental):
        (directory / "dataset_stats.json").write_text(
            json.dumps(stats),
            encoding="utf-8",
        )

    _write_config(
        baseline / "experiment_config.yaml",
        BASELINE_POLICIES,
        [0, 1_000],
    )
    _write_config(
        incremental / "experiment_config.yaml",
        ["SPARQ"],
        [1_000],
    )
    _write_raw(baseline / "raw" / "lru_cache_0.csv.gz")
    _write_raw(incremental / "raw" / "sparq_cache_1000.csv.gz")


def _write_config(path: Path, policies: list[str], sizes: list[int]) -> None:
    path.write_text(
        ExperimentConfig(
            policy=PolicyConfig(
                policies=policies,
                cache_sizes=sizes,
                include_unbounded_cache=False,
            )
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


def _summary_row(policy: str, cache_size: int, mode: str) -> dict[str, object]:
    return {
        "policy": policy,
        "cache_size": cache_size,
        "capacity_mode": mode,
        "hit_rate": 0.1,
        "mean_hit_response_cosine_distance": 0.2,
        "mean_hit_semantic_accuracy": 0.8,
        "mean_latency_ms": 10.0,
        "p95_latency_ms": 12.0,
        "p99_latency_ms": 14.0,
        "mean_policy_overhead_ms": 0.5,
        "policy_throughput_qps": 2_000.0,
        "sequential_end_to_end_throughput_qps": 100.0,
        "peak_process_rss_delta_mb": 5.0,
        "quality_adjusted_hit_rate@0.2": 0.08,
    }


def _write_raw(path: Path, *, response_id: str = "response-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "request_index": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "pair_index": 0,
                "prompt_id": "prompt-1",
                "response_id": response_id,
            }
        ]
    ).to_csv(path, index=False, compression="gzip")
