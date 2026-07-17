# EchoLLM W-SAGE

This branch adds **W-SAGE -- Windowed Semantic Admission and Gain-based
Eviction** to EchoLLM, plus reproducible OASST1 and WildChat-1M experiment
pipelines.

W-SAGE combines four mechanisms:

- a small LRU probation window that immediately admits new misses;
- a main cache selected by marginal, non-redundant semantic coverage;
- configurable soft coverage that prefers closer representatives while leaving
  the cache-hit predicate unchanged;
- separate recent and downsampled long-term demand evidence.

When replacement deltas tie, W-SAGE evicts the resident with the lowest
fractional semantic responsibility before using LRU as the final deterministic
tie-break.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-experiments.txt
```

Python 3.11 or newer is required.

## Use W-SAGE with EchoLLM

```python
from cache.sage import SAGESimilarityCache
from echollm import EchoLLM

cache = SAGESimilarityCache(
    max_size=1_000,
    hit_distance_threshold=0.18,
    prompt_embedder=my_embedding_function,
    window_fraction=0.05,
    soft_coverage=True,
    soft_coverage_power=1.0,
    recent_history_multiplier=4.0,
    long_history_multiplier=8.0,
    recent_evidence_weight=0.7,
)

echo = EchoLLM(cache=cache, llm=my_llm)
answer = echo.ask("How can I reset my password?")
```

`prompt_embedder` accepts a string and returns one numeric vector. Soft coverage
only changes admission and eviction scoring. A cache hit still requires the same
configured semantic-distance threshold used by every baseline.

Set `soft_coverage=False` for binary SAGE coverage. Set `window_fraction=0` for
the direct-admission SAGE ablation.

## Dataset construction

The default OASST1 loader combines train and validation, keeps every language,
and disables review/deletion filtering. It follows one deterministic
root-to-leaf path in each conversation tree. Every selected prompter appears
once and receives exactly one assistant response; sibling answers and branches
are not converted into repeated requests.

The WildChat loader extracts adjacent user-to-assistant turns, uses only the
current user message as the prompt, globally sorts by assistant completion time,
and retains the earliest 50,000 valid requests. Original model names are kept as
result metadata but are not added to cache keys.

## Run locally

Start Ollama, then run either configuration:

```bash
python -m _experiments.run \
  --config _experiments/configs/oasst1_default.yaml

python -m _experiments.run \
  --config _experiments/configs/wildchat_50k.yaml
```

Ollama generates each unique prompt once. Its real `LLMResponse`, including the
measured backend latency, is memoized and replayed identically across every
policy/capacity run.

## Run on Slurm

From the repository root:

```bash
git pull
sbatch _experiments/slurm/run_oasst1.sbatch
sbatch _experiments/slurm/run_wildchat_50k.sbatch
```

To run WildChat only after OASST1 succeeds:

```bash
OASST_JOB=$(sbatch --parsable _experiments/slurm/run_oasst1.sbatch)
sbatch --dependency="afterok:$OASST_JOB" \
  _experiments/slurm/run_wildchat_50k.sbatch
```

The scripts use the submission directory as the source checkout, copy a fresh
job-local worktree, start Ollama on a job-specific localhost port, verify CUDA,
and preserve results outside the temporary copy. Override the backend model if
needed:

```bash
MODEL=llama3.2:1b sbatch _experiments/slurm/run_oasst1.sbatch
```

Ollama's repetitive `[GIN]` request lines are filtered from its log. The Slurm
output contains LLM-generation progress, one tqdm request bar per
policy/capacity experiment, each experiment's duration, and total pipeline/job
duration.

## Results

Each run writes:

```text
<results>/<run-name>/
|-- config.json
|-- dataset_stats.json
|-- summary.csv
|-- summary.json
|-- raw/*.csv.gz
`-- plots/*.png
```

Raw results are first written as `.partial` files and published atomically only
after a policy run completes.

The summaries include hit rate, semantic accuracy, quality-adjusted hit rate,
good-hit precision, bad-hit rate, mean/p95/p99 latency, policy overhead,
throughput, CPU time, and process RSS context. Raw SAGE rows also expose whether
the incoming miss entered probation and whether the resident leaving probation
was promoted to the main cache.

## Validation

```bash
PYTHONPATH=. pytest -q
```

See [_experiments/README.md](_experiments/README.md) for the full methodology and
configuration reference.
