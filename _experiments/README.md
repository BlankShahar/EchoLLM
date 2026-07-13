# OASST1 semantic-cache experiments

## Research question

Does SAGE improve semantic cache hit rate and latency over basic replacement policies without obtaining those gains by returning unsuitable cached answers?

## Dataset extraction

The loader reads OASST1 message rows and creates context-free, direct English
prompter-to-assistant pairs. With `response_selection: top_rank`, the
lowest-ranked direct assistant response is used as the reference; quality label
and source order are deterministic fallbacks.

Using one selected answer per prompt is important. With all valid replies, the exact same prompt can have several acceptable references, making a cache hit appear wrong merely because it returned a different valid reply.

Local JSONL and Parquet inputs are also supported for offline reproducibility.

## Workload modes

- `dataset_order`: repeatedly walks the extracted pair sequence.
- `chronological`: sorts prompts by `created_date`; with `request_count: null`, each selected prompt appears exactly once.
- `shuffled`: repeats independently shuffled passes.
- `zipf_clustered`: clusters prompt embeddings and samples semantic topics with Zipf-skewed popularity, then samples a prompt inside the selected topic.

The default trace is `chronological` over the combined train and validation
splits of the complete selected English dataset, with no synthetic repetition
and no warm-up exclusion. Other modes remain available for controlled workload
studies.

The capacity sweep accepts zero. `include_unbounded_cache: true` adds a final
capacity equal to the number of unique prompt strings in the actual trace, which
is sufficient to avoid capacity eviction. Every policy/capacity run constructs
a fresh cache. The experiment baselines and SAGE are in-memory and do not use
EchoLLM's FAISS or response SQLite databases; the shared SQLite embedding cache
contains only model/text embeddings and is intentionally reused.

## Fair baselines

LRU, LFU, FIFO, random replacement, and SAGE all use:

- exact resident-vector scans;
- the same prompt embeddings;
- cosine distance;
- the same hit threshold;
- the same capacity measured in entries;
- the same nearest covering response selection;
- the same trace and random seed.

This prevents approximate-nearest-neighbor recall from becoming a hidden policy variable.

## Response correctness / anti-reward-hacking metrics

On every hit, the returned cached answer and that request's OASST1 reference answer are embedded using the **quality evaluator model**, which is separately configurable from the prompt-matching model.

Let \(d_r\) be their cosine distance.

### Hit-only mean response distance

\[
\frac{1}{|H|}\sum_{q\in H}d_r(q)
\]

This measures the average correctness of served hits, but it is not sufficient alone: a very conservative policy can achieve a low value by serving almost no hits.

### Quality-adjusted hit rate at threshold \(t\)

\[
\frac{|\{q: q\text{ is a hit and }d_r(q)\le t\}|}{|Q|}
\]

This is the preferred joint metric. It rewards a policy only for hits whose returned response is close enough to the actual reference response.

### Bad-hit rate at threshold \(t\)

\[
\frac{|\{q: q\text{ is a hit and }d_r(q)>t\}|}{|Q|}
\]

This exposes policies that improve hit rate by using an overly broad semantic threshold.

### End-to-end response distance

This is the mean returned-vs-reference distance over all requests. On a miss,
the returned response is the generated LLM response; on a hit, it is the cached
response. Always report it with hit rate and the hit-only quality metrics.

## Latency

OASST1 does not contain serving latency. The runner uses EchoLLM's existing
`ILLM` interface: `Ollama.ask()` measures the real generation call and returns
an `LLMResponse` containing both text and latency. A small in-memory `ILLM`
decorator generates every prompt once, then replays the identical response and
latency to every policy.

Policy runtime is measured separately. Response-quality embedding distance is
outside the policy timer: a hit measures lookup only, while a miss measures
lookup plus admission.

### Throughput and resources

- `policy_throughput_qps`: measured requests divided by total lookup/admission seconds.
- `sequential_end_to_end_throughput_qps`: `1000 / mean_latency_ms`, assuming serial request handling.
- `runner_throughput_qps`: actual measured trace-processing rate, including metrics and raw-output work.
- `peak_process_rss_mb`: sampled peak resident memory for the Python process.
- `peak_process_rss_delta_mb`: peak RSS increase from the start of the measured policy trace.
- `runner_cpu_time_seconds`: process CPU time during the measured trace.

GPU utilization is not a replacement-policy metric: Ollama generation and
embedding computation use the GPU, while exact cache lookup and SAGE scoring
are NumPy/CPU operations.

## Configuration

Edit `configs/oasst1_default.yaml`.

Important fields:

```yaml
embedding:
  prompt_model_name: sentence-transformers/all-MiniLM-L6-v2
  quality_model_name: sentence-transformers/all-mpnet-base-v2

llm:
  provider: ollama
  model: llama3.2:1b
  host: http://127.0.0.1:11434
  options:
    num_predict: 256

trace:
  mode: chronological
  request_count: null

policy:
  cache_sizes: [0, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
  include_unbounded_cache: true
  hit_distance_threshold: 0.18
  sage_ghost_capacity: 4096
  sage_decay_half_life_requests: 10000

quality:
  good_hit_distance_thresholds: [0.1, 0.2, 0.3]
```

The numerical thresholds are experiment parameters, not universal constants. Tune the prompt hit threshold on a validation split and freeze it before test comparison. Likewise, define answer-quality thresholds from a held-out sample or human-judged calibration set.

## Run

```bash
PYTHONPATH=. python -m _experiments.run \
  --config _experiments/configs/oasst1_default.yaml
```

For the cluster workflow:

```bash
sbatch _experiments/slurm/run_oasst1.sbatch
```

The script copies a fresh worktree, starts and verifies Ollama on CUDA, runs the
framework-native experiment with sentence-transformer embeddings on CUDA, and
keeps results outside the temporary code copy.

## Raw output schema

Each compressed raw CSV includes:

- request and pair identifiers;
- warm-up/measured flag;
- policy and capacity;
- hit/miss;
- matched-prompt distance;
- returned-vs-reference response distance;
- backend latency returned by `ILLM.ask()`;
- measured policy overhead;
- total end-to-end latency;
- SAGE admission decision and net delta where applicable.

## Suggested paper-quality protocol

1. Choose train/validation/test partitions without leaking identical message trees across splits.
2. Calibrate semantic-hit and answer-quality thresholds only on validation data.
3. Report hit rate, response quality, mean/p95/p99 latency, throughput, overhead, and peak RSS together.
4. Sweep cache capacity and ghost capacity.
5. Include SAGE ablations when making claims about which SAGE component caused an improvement.
6. Inspect a stratified sample of good and bad hits manually or with a separate judged benchmark.
