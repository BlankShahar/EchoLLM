# Feature-branch file map

## EchoLLM integration

- `cache/models.py`: Pydantic single-pass lookup result.
- `cache/icache.py`: backward-compatible `lookup()` contract.
- `echollm/echollm.py`: lookup-context propagation and separate cache/backend timing.
- `cache/sage/`: W-SAGE policy, configuration, evidence, scorer, models, and storage.
- `cache/sage_similarity_cache.py`: flat-layout compatibility import.

## Experiment project

- `_experiments/config.py`: discriminated dataset and policy configuration.
- `_experiments/datasets.py`: dataset-source loading and dispatch.
- `_experiments/oasst1.py`: deterministic single-path OASST1 extraction.
- `_experiments/wildchat.py`: bounded-memory chronological WildChat extraction.
- `_experiments/trace.py`: shared trace construction.
- `_experiments/runner.py`: full-grid or single-index policy replay and metrics.
- `_experiments/aggregate.py`: validates and combines Slurm-array task outputs.
- `_experiments/configs/oasst1_default.yaml`: OASST1 W-SAGE experiment.
- `_experiments/configs/wildchat_15k.yaml`: first-15K WildChat experiment.
- `_experiments/slurm/run_experiment.sh`: shared cluster runtime.
- `_experiments/slurm/run_oasst1.sbatch`: 50-task OASST1 array.
- `_experiments/slurm/run_wildchat_15k.sbatch`: 50-task WildChat array.
- `_experiments/slurm/submit_*_array.sh`: array plus dependent aggregation submission.

## Validation

- `tests/test_sage_*`: W-SAGE math, policy, evidence, persistence, and rollback.
- `tests/test_oasst1_loader.py`: single-path selection.
- `tests/test_wildchat_loader.py`: chronological cap and malformed-row handling.
- `tests/test_experiment_runner.py`: progress, timing, raw metadata, and atomic output.
