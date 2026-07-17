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
and retains the earliest 15,000 valid requests. Original model names are kept as
result metadata but are not added to cache keys.

## Run locally

Start Ollama, then run either configuration:

```bash
python -m _experiments.run \
  --config _experiments/configs/oasst1_default.yaml

python -m _experiments.run \
  --config _experiments/configs/wildchat_15k.yaml
```

Every trace request is sent through `EchoLLM`. A cache miss always calls Ollama,
even when the exact prompt appeared earlier in the trace; only a cache hit skips
the backend. The `LLMResponse.latency` from that live call is recorded for the
request. Response embeddings used only for quality evaluation may be reused and
do not affect cache behavior or the measured backend/cache latency.

## Run on Slurm

From the repository root:

```bash
git pull
MAX_CONCURRENT=8 bash _experiments/slurm/submit_both_arrays.sh
```

Or submit one dataset:

```bash
MAX_CONCURRENT=8 bash _experiments/slurm/submit_oasst1_array.sh
MAX_CONCURRENT=8 bash _experiments/slurm/submit_wildchat_15k_array.sh
```

Each dataset is a 50-task job array: one full trace replay for each of five
policies and ten capacities, including unbounded. `MAX_CONCURRENT` limits the
number of GPUs used simultaneously per dataset. A dependent CPU job validates
all tasks, combines their results, and generates the final plots. Override the
backend model if needed:

```bash
MODEL=llama3.2:1b MAX_CONCURRENT=16 \
  bash _experiments/slurm/submit_oasst1_array.sh
```

Every task has an isolated worktree, cache, Ollama port, log directory, and
result directory. Ollama's repetitive `[GIN]` lines are filtered. Task logs use
`%A_%a`, for example `echollm-wsage-oasst1-12345_7.out`.

## Results

Each run writes:

```text
<results>/<run-name>/
|-- experiment_config.yaml
|-- dataset_stats.json
|-- summary.csv
|-- summary.json
|-- raw/*.csv.gz
|-- plots/*.png
|-- aggregation.json
`-- tasks/task-*/summary.csv
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
