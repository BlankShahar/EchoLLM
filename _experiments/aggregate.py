import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .plotting import generate_plots


def aggregate_results(
    tasks_root: Path,
    output_directory: Path,
    *,
    expected_tasks: int,
    config_path: Path | None = None,
) -> Path:
    task_directories = sorted(path.parent for path in tasks_root.glob("*/summary.csv"))
    if len(task_directories) != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} completed task summaries under {tasks_root}, "
            f"found {len(task_directories)}"
        )

    frames: list[pd.DataFrame] = []
    for task_directory in task_directories:
        frame = pd.read_csv(task_directory / "summary.csv")
        if len(frame) != 1:
            raise RuntimeError(
                f"Array task {task_directory.name} produced {len(frame)} summaries; "
                "expected exactly one"
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    identity_columns = ["policy", "cache_size", "capacity_mode"]
    duplicates = combined.duplicated(identity_columns, keep=False)
    if duplicates.any():
        repeated = combined.loc[duplicates, identity_columns].to_dict("records")
        raise RuntimeError(f"Duplicate experiment grid entries: {repeated}")

    combined = combined.sort_values(["cache_size", "policy"]).reset_index(drop=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_directory / "summary.csv", index=False)
    (output_directory / "summary.json").write_text(
        json.dumps(combined.to_dict("records"), indent=2),
        encoding="utf-8",
    )

    _merge_raw_results(task_directories, output_directory / "raw")
    _merge_dataset_stats(task_directories, output_directory / "dataset_stats.json")
    if config_path is not None:
        shutil.copy2(config_path, output_directory / "experiment_config.yaml")

    (output_directory / "aggregation.json").write_text(
        json.dumps(
            {
                "expected_tasks": expected_tasks,
                "completed_tasks": len(task_directories),
                "summary_rows": len(combined),
                "task_directories": [path.name for path in task_directories],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    generate_plots(output_directory)
    return output_directory


def _merge_raw_results(task_directories: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for task_directory in task_directories:
        for source in (task_directory / "raw").glob("*.csv.gz"):
            target = destination / source.name
            if target.exists():
                raise RuntimeError(f"Duplicate raw result file: {target.name}")
            source.replace(target)


def _merge_dataset_stats(task_directories: list[Path], destination: Path) -> None:
    payloads = [
        (task_directory / "dataset_stats.json").read_bytes()
        for task_directory in task_directories
    ]
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise RuntimeError("Array tasks used inconsistent dataset traces")
    destination.write_bytes(payloads[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate EchoLLM Slurm array results")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args()
    output = aggregate_results(
        arguments.tasks_root,
        arguments.output_dir,
        expected_tasks=arguments.expected_tasks,
        config_path=arguments.config,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
