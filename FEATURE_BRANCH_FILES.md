# Feature-branch file map

## EchoLLM integration

- `cache/models.py`: Pydantic single-pass lookup result.
- `cache/icache.py`: backward-compatible `lookup()` contract.
- `echollm/echollm.py`: lookup-context propagation and separate cache/backend timing.
- `cache/sage/`: W-SAGE policy, configuration, evidence, scorer, models, and storage.
- `cache/sage_similarity_cache.py`: flat-layout compatibility import.
- `cache/sparq/`: SPARQ configuration, Pydantic models, and policy implementation.
- `cache/sparq_similarity_cache.py`: flat-layout compatibility import.

## Experiment project

- `_experiments/config.py`: discriminated dataset and policy configuration.
- `_experiments/datasets.py`: dataset-source loading and dispatch.
- `_experiments/oasst1.py`: deterministic single-path OASST1 extraction.
- `_experiments/wildchat.py`: bounded-memory chronological WildChat extraction.
- `_experiments/trace.py`: shared trace construction.
- `_experiments/runner.py`: full-grid or single-index policy replay and metrics.
- `_experiments/aggregate.py`: validates and combines Slurm-array task outputs.
- `_experiments/merge_results.py`: validates and merges SPARQ-only policies or
  supplemental all-policy capacity ranges, then regenerates all-policy plots.
- `_experiments/configs/oasst1_default.yaml`: OASST1 SAGE/SPARQ comparison.
- `_experiments/configs/wildchat_15k.yaml`: first-15K WildChat SAGE/SPARQ comparison.
- `_experiments/slurm/run_experiment.sh`: shared cluster runtime.
- `_experiments/slurm/run_oasst1.sbatch`: 91-task OASST1 array.
- `_experiments/slurm/run_wildchat_15k.sbatch`: 91-task WildChat array.
- `_experiments/slurm/submit_*_array.sh`: array plus dependent aggregation submission.
- `_experiments/slurm/submit_*_sparq_only.sh`: incremental SPARQ-only CPU replay.
- `_experiments/slurm/aggregate_incremental_policy.sbatch`: verified merge and
  all-policy plot generation.
- `_experiments/slurm/submit_*_sub1k.sh`: all-policy supplemental sub-1K replay.
- `_experiments/slurm/aggregate_incremental_capacities.sbatch`: verified
  capacity-grid merge and plot generation.

## Validation

- `tests/test_sage_*`: W-SAGE math, policy, evidence, persistence, and rollback.
- `tests/test_sparq_cache.py`: SPARQ credit, probation, promotion, aging, and flow.
- `tests/test_oasst1_loader.py`: single-path selection.
- `tests/test_wildchat_loader.py`: chronological cap and malformed-row handling.
- `tests/test_experiment_runner.py`: progress, timing, raw metadata, and atomic output.
- `tests/test_incremental_merge.py`: grid filtering, compatibility checks, trace
  fingerprints, and endpoint merging.
