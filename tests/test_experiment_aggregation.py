import json
from pathlib import Path

import pandas as pd

from _experiments.aggregate import aggregate_results


def test_aggregate_combines_one_result_per_task_and_moves_raw_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-000", "LRU", 0, "no_cache")
    _write_task(tasks / "task-001", "SAGE", 10, "bounded")
    config = tmp_path / "config.yaml"
    config.write_text("dataset: test\n", encoding="utf-8")
    monkeypatch.setattr("_experiments.aggregate.generate_plots", lambda _: [])

    output = aggregate_results(
        tasks,
        tmp_path / "combined",
        expected_tasks=2,
        config_path=config,
    )

    summary = pd.read_csv(output / "summary.csv")
    assert list(summary["policy"]) == ["LRU", "SAGE"]
    assert (output / "raw" / "lru_cache_0.csv.gz").exists()
    assert (output / "raw" / "sage_cache_10.csv.gz").exists()
    assert json.loads((output / "aggregation.json").read_text())["summary_rows"] == 2


def test_aggregate_rejects_missing_tasks(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-000", "LRU", 0, "no_cache")

    try:
        aggregate_results(tasks, tmp_path / "combined", expected_tasks=2)
    except RuntimeError as error:
        assert "Expected 2 completed task summaries" in str(error)
    else:
        raise AssertionError("Expected aggregation to reject an incomplete array")


def test_aggregate_expands_one_shared_no_cache_row_for_all_policies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks / "task-000", "LRU", 0, "no_cache")
    _write_task(tasks / "task-001", "SAGE", 10, "bounded")
    config = tmp_path / "config.yaml"
    config.write_text(
        "policy:\n  policies: [LRU, LFU, FIFO, RR, SAGE]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("_experiments.aggregate.generate_plots", lambda _: [])

    output = aggregate_results(
        tasks,
        tmp_path / "combined",
        expected_tasks=2,
        config_path=config,
    )

    summary = pd.read_csv(output / "summary.csv")
    zero = summary.loc[summary["capacity_mode"] == "no_cache"]
    assert set(zero["policy"]) == {"LRU", "LFU", "FIFO", "RR", "SAGE"}
    assert len(list((output / "raw").glob("*cache_0.csv.gz"))) == 1


def _write_task(
    directory: Path,
    policy: str,
    cache_size: int,
    capacity_mode: str,
) -> None:
    raw = directory / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "policy": policy,
                "cache_size": cache_size,
                "capacity_mode": capacity_mode,
                "hit_rate": 0.0,
            }
        ]
    ).to_csv(directory / "summary.csv", index=False)
    (directory / "dataset_stats.json").write_text('{"trace_requests": 2}')
    (raw / f"{policy.lower()}_cache_{cache_size}.csv.gz").write_bytes(b"raw")
