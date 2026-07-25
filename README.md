# EchoLLM semantic cache policies

This branch adds two semantic admission and eviction policies to EchoLLM:

- **W-SAGE**, which optimizes counterfactual marginal semantic coverage.
- **SPARQ**, a compact semantic-frequency policy with aging and an LRU
  probation queue.

It also includes reproducible OASST1 and WildChat-1M experiment pipelines.

SPARQ assigns proximity-weighted credit to cached prompts that cover each
request. Close representatives receive more credit, overlapping representatives
split the same demand, and stale scores are periodically aged. New misses first
enter a small LRU probation queue; a probation resident enters the main cache
only when its score beats the weakest main-cache resident.

W-SAGE combines five mechanisms:

- a configurable LRU probation window that immediately admits new misses;
- a main cache selected by marginal, non-redundant semantic coverage;
- configurable soft coverage that prefers closer representatives while leaving
  the cache-hit predicate unchanged;
- separate recent and downsampled long-term demand evidence;
- an optional direct-frequency regularizer for near-equivalent coverage choices.

When replacement deltas tie, W-SAGE evicts the resident with the lowest
fractional semantic responsibility before using LRU as the final deterministic
tie-break.

The frequency regularizer uses the same resident set and capacity. It does not
maintain a shadow LFU cache: it only protects directly reused residents when
their normalized popularity outweighs a very small marginal-coverage
difference.

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

## Use SPARQ with EchoLLM

```python
from cache.sparq import SPARQSimilarityCache
from echollm import EchoLLM

cache = SPARQSimilarityCache(
    max_size=1_000,
    hit_distance_threshold=0.25,
    prompt_embedder=my_embedding_function,
    window_fraction=0.20,
    credit_power=2.0,
    aging_factor=0.5,
)

echo = EchoLLM(cache=cache, llm=my_llm)
answer = echo.ask("How can I reset my password?")
```

The default aging interval is four times the cache capacity. Set
`aging_interval_requests` explicitly to override it. Setting
`window_fraction=0` produces the direct semantic-frequency ablation.

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

Every trace request is sent through `EchoLLM`. The default Slurm workflow first
calls Ollama once per unique prompt and records its response together with the
real `LLMResponse.latency`. Policy tasks then call that recorded `ILLM` on every
cache miss, so cache behavior is unchanged while all policies see identical,
deterministic backend output and latency. Response-quality embeddings are
outside the measured backend/cache latency.

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

Each current dataset configuration is a 91-task array: one shared no-cache run,
then 15 positive capacities for six policies. `MAX_CONCURRENT` limits
simultaneous replay tasks. One GPU preparation job materializes the exact trace,
records backend responses, and computes embeddings once. The policy grid then
runs as CPU-only replay jobs; a dependent CPU job validates all tasks, combines
the results, and generates the final plots.

The full sweep runs from 1,000 entries through 15,000 entries. Its upper points
are intentionally near effective-unbounded behavior and should be interpreted as
convergence checks rather than as capacity-pressure comparisons. Both traces
default to `qwen3:4b-instruct` with at most 64 generated tokens.

### Add SPARQ to completed benchmark results

Do not rerun the five existing policies or Ollama. Point the incremental
workflow at the two completed aggregate directories:

```bash
export OASST_BASELINE_RESULTS="$HOME/_experiments/echollm-sage/results/oasst1-wsage-19530834"
export WILDCHAT_BASELINE_RESULTS="$HOME/_experiments/echollm-sage/results/wildchat15k-wsage-19530847"

MAX_CONCURRENT=8 \
  bash _experiments/slurm/submit_both_sparq_only.sh
```

Each trace submits one CPU replay task for every capacity marked `bounded` in
the baseline. For the current results this is 14 tasks, covering 1,000 through
14,000; the policy-independent 15,000 endpoint is copied during aggregation.
The script derives this plan automatically and reuses the baseline's recorded
LLM database, prepared trace, and embedding cache. A dependent merge job
validates the configurations, capacity grid, dataset statistics, and raw
request-trace fingerprint before regenerating all-policy plots.

The submission output prints both job IDs and exact paths. Final merged results
are written below:

```text
$HOME/_experiments/echollm-sage/results/<trace>-sparq-incremental-<array-job-id>/comparison/
├── summary.csv
├── summary.json
├── merge_manifest.json
└── plots/
```

If automatic artifact discovery cannot find an older prepared trace, explicitly
set `RECORDED_LLM_PATH`, `PREPARED_PAIRS_PATH`, and
`EMBEDDING_CACHE_PATH` to the artifacts used by that baseline.

Preview the runtime projection without submitting:

```bash
DRY_RUN=1 MAX_CONCURRENT=8 \
  bash _experiments/slurm/submit_wildchat_15k_array.sh
```

The projection covers execution after Slurm allocates the one preparation GPU
and the replay CPUs; queue delay cannot be guaranteed by the scripts.

Override the backend model if needed:

```bash
MODEL=qwen3:4b-instruct MAX_CONCURRENT=8 \
  bash _experiments/slurm/submit_oasst1_array.sh
```

Every task has an isolated policy instance, worktree, log directory, and result
directory. The prepared trace, recorded backend, and embedding database are
immutable shared inputs; no cache-policy state is shared between tasks.
Ollama's repetitive `[GIN]` lines are filtered. Replay logs use `%A_%a`, for
example `echollm-cache-replay-12345_7.out`.
The submission helper prints exact `squeue`, `sacct`, `tail`, `scancel`, and
result-directory commands containing the assigned job IDs.

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
throughput, CPU time, and process RSS context. Raw SAGE and SPARQ rows also
expose admission/promotion outcomes. SPARQ rows include the candidate and victim
scores used by the replacement decision.

## Validation

```bash
PYTHONPATH=. pytest -q
```

See [_experiments/README.md](_experiments/README.md) for the full methodology and
configuration reference.
