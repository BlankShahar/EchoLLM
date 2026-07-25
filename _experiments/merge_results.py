import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import ExperimentConfig
from .plotting import generate_plots


_IDENTITY_COLUMNS = ["policy", "cache_size", "capacity_mode"]
_REQUIRED_BASELINE_POLICIES = {"LRU", "LFU", "FIFO", "RR", "SAGE"}
_TRACE_COLUMNS = [
    "request_index",
    "created_at",
    "pair_index",
    "prompt_id",
    "response_id",
]


def validate_incremental_setup(
    baseline_directory: Path,
    source_config_path: Path,
    *,
    policy: str = "SPARQ",
) -> list[int]:
    """Fail before submission when the proposed replay cannot be merged fairly."""
    baseline = _read_summary(baseline_directory)
    _validate_baseline_policies(baseline)
    source_config = ExperimentConfig.from_yaml(source_config_path).select_grid(
        policies=[policy],
        positive_cache_sizes_only=True,
    )
    baseline_config = ExperimentConfig.from_yaml(
        _published_config_path(baseline_directory)
    )
    _assert_compatible_configs(
        baseline_config,
        source_config,
        compare_artifact_paths=False,
    )
    baseline_capacities = _baseline_bounded_capacities(baseline)
    source_capacities = set(source_config.policy.cache_sizes)
    unavailable = baseline_capacities - source_capacities
    if unavailable:
        raise RuntimeError(
            "Baseline bounded capacities are absent from the source config: "
            f"{sorted(unavailable)}"
        )
    return sorted(baseline_capacities)


def merge_policy_results(
    baseline_directory: Path,
    incremental_directory: Path,
    output_directory: Path,
    *,
    policy: str = "SPARQ",
) -> Path:
    """Merge one incremental policy with a completed baseline benchmark."""
    baseline_directory = baseline_directory.resolve()
    incremental_directory = incremental_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory in {baseline_directory, incremental_directory}:
        raise ValueError("Merged output must be separate from both source directories")

    baseline = _read_summary(baseline_directory)
    incremental = _read_summary(incremental_directory)
    _validate_baseline_policies(baseline)
    incremental_policies = set(incremental["policy"].astype(str))
    if incremental_policies != {policy}:
        raise RuntimeError(
            f"Incremental results must contain only {policy}; "
            f"found {sorted(incremental_policies)}"
        )
    if set(incremental["capacity_mode"].astype(str)) != {"bounded"}:
        raise RuntimeError("Incremental results must contain bounded capacities only")

    baseline_stats = _read_json(baseline_directory / "dataset_stats.json")
    incremental_stats = _read_json(incremental_directory / "dataset_stats.json")
    if baseline_stats != incremental_stats:
        raise RuntimeError("Baseline and incremental runs used different dataset traces")

    baseline_config = ExperimentConfig.from_yaml(
        _effective_config_path(baseline_directory)
    )
    incremental_config = ExperimentConfig.from_yaml(
        _effective_config_path(incremental_directory)
    )
    _assert_compatible_configs(
        baseline_config,
        incremental_config,
        compare_artifact_paths=True,
    )

    incremental_capacities = set(
        incremental.loc[
            incremental["capacity_mode"] == "bounded",
            "cache_size",
        ].astype(int)
    )
    _assert_matching_bounded_capacities(baseline, incremental_capacities)

    baseline_fingerprint = _trace_fingerprint(baseline_directory / "raw")
    incremental_fingerprint = _trace_fingerprint(incremental_directory / "raw")
    if baseline_fingerprint != incremental_fingerprint:
        raise RuntimeError("Raw results contain different request traces")

    baseline_without_policy = baseline.loc[baseline["policy"] != policy].copy()
    endpoints = _policy_independent_endpoints(baseline_without_policy, policy)
    combined = pd.concat(
        [baseline_without_policy, incremental, endpoints],
        ignore_index=True,
    )
    duplicates = combined.duplicated(_IDENTITY_COLUMNS, keep=False)
    if duplicates.any():
        repeated = combined.loc[duplicates, _IDENTITY_COLUMNS].to_dict("records")
        raise RuntimeError(f"Duplicate merged grid entries: {repeated}")
    combined = combined.sort_values(["cache_size", "policy"]).reset_index(drop=True)

    output_directory.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_directory / "summary.csv", index=False)
    (output_directory / "summary.json").write_text(
        json.dumps(combined.to_dict("records"), indent=2),
        encoding="utf-8",
    )
    shutil.copy2(
        baseline_directory / "dataset_stats.json",
        output_directory / "dataset_stats.json",
    )
    shutil.copy2(
        _published_config_path(baseline_directory),
        output_directory / "baseline_config.yaml",
    )
    shutil.copy2(
        _published_config_path(incremental_directory),
        output_directory / "incremental_config.yaml",
    )
    manifest = {
        "policy": policy,
        "baseline_directory": str(baseline_directory),
        "incremental_directory": str(incremental_directory),
        "summary_rows": len(combined),
        "bounded_capacities": sorted(incremental_capacities),
        "trace_fingerprint": baseline_fingerprint,
        "raw_results": {
            "baseline": str(baseline_directory / "raw"),
            "incremental": str(incremental_directory / "raw"),
        },
    }
    (output_directory / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    generate_plots(output_directory)
    return output_directory


def _read_summary(directory: Path) -> pd.DataFrame:
    summary_path = directory / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    frame = pd.read_csv(summary_path)
    missing = set(_IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Summary is missing identity columns: {sorted(missing)}")
    duplicates = frame.duplicated(_IDENTITY_COLUMNS, keep=False)
    if duplicates.any():
        repeated = frame.loc[duplicates, _IDENTITY_COLUMNS].to_dict("records")
        raise RuntimeError(f"Source summary has duplicate grid entries: {repeated}")
    return frame


def _validate_baseline_policies(baseline: pd.DataFrame) -> None:
    policies = set(baseline["policy"].astype(str))
    missing = _REQUIRED_BASELINE_POLICIES - policies
    if missing:
        raise RuntimeError(
            f"Baseline results are missing policies: {sorted(missing)}"
        )


def _assert_matching_bounded_capacities(
    baseline: pd.DataFrame,
    expected: set[int],
) -> None:
    actual_by_policy = _bounded_capacities_by_policy(baseline)
    for policy, actual in actual_by_policy.items():
        if actual != expected:
            raise RuntimeError(
                f"{policy} bounded capacities {sorted(actual)} do not match "
                f"the incremental capacities {sorted(expected)}"
            )


def _baseline_bounded_capacities(baseline: pd.DataFrame) -> set[int]:
    actual_by_policy = _bounded_capacities_by_policy(baseline)
    reference_policy = sorted(actual_by_policy)[0]
    reference = actual_by_policy[reference_policy]
    for policy, actual in actual_by_policy.items():
        if actual != reference:
            raise RuntimeError(
                f"Baseline bounded capacities differ: {reference_policy} has "
                f"{sorted(reference)}, while {policy} has {sorted(actual)}"
            )
    if not reference:
        raise RuntimeError("Baseline contains no bounded cache capacities")
    return reference


def _bounded_capacities_by_policy(
    baseline: pd.DataFrame,
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for policy in _REQUIRED_BASELINE_POLICIES:
        result[policy] = set(
            baseline.loc[
                (baseline["policy"] == policy)
                & (baseline["capacity_mode"] == "bounded"),
                "cache_size",
            ].astype(int)
        )
    return result


def _assert_compatible_configs(
    baseline: ExperimentConfig,
    incremental: ExperimentConfig,
    *,
    compare_artifact_paths: bool,
) -> None:
    baseline_view = _comparison_config(
        baseline,
        compare_artifact_paths=compare_artifact_paths,
    )
    incremental_view = _comparison_config(
        incremental,
        compare_artifact_paths=compare_artifact_paths,
    )
    if baseline_view != incremental_view:
        differences = [
            key
            for key in baseline_view
            if baseline_view[key] != incremental_view[key]
        ]
        raise RuntimeError(
            "Baseline and incremental experiment settings differ in: "
            + ", ".join(differences)
        )


def _comparison_config(
    config: ExperimentConfig,
    *,
    compare_artifact_paths: bool,
) -> dict[str, Any]:
    llm = {
        "model": config.llm.model,
        "options": config.llm.options,
    }
    embedding = {
        "prompt_model_name": config.embedding.prompt_model_name,
        "quality_model_name": config.embedding.quality_model_name,
    }
    if compare_artifact_paths:
        llm["provider"] = config.llm.provider.value
        llm["recorded_path"] = (
            str(config.llm.recorded_path.resolve())
            if config.llm.recorded_path is not None
            else None
        )
        embedding["cache_path"] = str(config.embedding.cache_path.resolve())
    return {
        "dataset": config.dataset.model_dump(mode="json"),
        "trace": config.trace.model_dump(mode="json"),
        "embedding": embedding,
        "llm": llm,
        "hit_distance_threshold": config.policy.hit_distance_threshold,
        "quality": config.quality.model_dump(mode="json"),
        "resources": config.resources.model_dump(mode="json"),
    }


def _policy_independent_endpoints(
    baseline: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    endpoints = baseline.loc[
        baseline["capacity_mode"].isin({"no_cache", "unbounded"})
    ]
    copies: list[pd.Series] = []
    for _, group in endpoints.groupby(["cache_size", "capacity_mode"], sort=False):
        copy = group.iloc[0].copy()
        copy["policy"] = policy
        copies.append(copy)
    if not copies:
        return baseline.iloc[0:0].copy()
    return pd.DataFrame(copies)


def _trace_fingerprint(raw_directory: Path) -> str:
    paths = sorted(raw_directory.glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No raw result files found under {raw_directory}")
    preferred = [path for path in paths if "cache_0.csv.gz" in path.name]
    source = preferred[0] if preferred else paths[0]
    available = set(pd.read_csv(source, nrows=0).columns)
    columns = [column for column in _TRACE_COLUMNS if column in available]
    if columns != _TRACE_COLUMNS:
        missing = set(_TRACE_COLUMNS) - set(columns)
        raise RuntimeError(
            f"Raw trace {source} is missing identity columns: {sorted(missing)}"
        )
    trace = pd.read_csv(source, usecols=columns, dtype=str).fillna("")
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    for row in trace.itertuples(index=False, name=None):
        digest.update(b"\x1e")
        digest.update("\x1f".join(row).encode("utf-8"))
    return digest.hexdigest()


def _published_config_path(directory: Path) -> Path:
    path = directory / "experiment_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Published experiment config not found: {path}")
    return path


def _effective_config_path(directory: Path) -> Path:
    task_configs = sorted((directory / "tasks").glob("task-*/config.json"))
    return task_configs[0] if task_configs else _published_config_path(directory)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or merge an incremental EchoLLM policy benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--baseline-dir", type=Path, required=True)
    validate.add_argument("--source-config", type=Path, required=True)
    validate.add_argument("--policy", default="SPARQ")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--baseline-dir", type=Path, required=True)
    merge.add_argument("--incremental-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--policy", default="SPARQ")

    arguments = parser.parse_args()
    if arguments.command == "validate":
        capacities = validate_incremental_setup(
            arguments.baseline_dir,
            arguments.source_config,
            policy=arguments.policy,
        )
        print("Incremental experiment is compatible with the baseline.", flush=True)
        print(
            "CACHE_SIZES=" + ",".join(str(size) for size in capacities),
            flush=True,
        )
        return
    output = merge_policy_results(
        arguments.baseline_dir,
        arguments.incremental_dir,
        arguments.output_dir,
        policy=arguments.policy,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
