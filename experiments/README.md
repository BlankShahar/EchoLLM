# OASST1 semantic-cache experiments

## Research question

Does SAGE improve semantic cache hit rate and latency over basic replacement policies without obtaining those gains by returning unsuitable cached answers?

## Dataset extraction

The loader reads OASST1 message rows and creates direct English prompter-to-assistant pairs. Deleted/rejected rows are excluded. With `response_selection: top_rank`, the lowest-ranked direct assistant response is used as the reference; quality label and source order are deterministic fallbacks.

Using one selected answer per prompt is important. With all valid replies, the exact same prompt can have several acceptable references, making a cache hit appear wrong merely because it returned a different valid reply.

Local JSONL and Parquet inputs are also supported for offline reproducibility.

## Workload modes

- `dataset_order`: repeatedly walks the extracted pair sequence.
- `shuffled`: repeats independently shuffled passes.
- `zipf_clustered`: clusters prompt embeddings and samples semantic topics with Zipf-skewed popularity, then samples a prompt inside the selected topic.

The default `zipf_clustered` trace introduces both topic-level reuse and paraphrase variation—the intended semantic-cache workload. A prefix of the trace can warm all policies; warm-up requests still mutate cache state but do not enter reported metrics.

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

Misses use the reference response and therefore contribute zero distance. This represents delivered-output error under the simulator, but it can be minimized by never hitting. Always report it with hit rate and quality-adjusted hit rate.

## Latency

OASST1 does not contain serving latency. The harness therefore reports **simulated latency**, using either:

- a fixed miss latency; or
- a configurable linear approximation based on prompt and response word counts.

Policy runtime is measured separately. For a production trace with observed latency, replace `LatencyModel` with a measured-field implementation; no policy code needs to change.

## Configuration

Edit `configs/oasst1_default.yaml`.

Important fields:

```yaml
embedding:
  prompt_model_name: sentence-transformers/all-MiniLM-L6-v2
  quality_model_name: sentence-transformers/all-mpnet-base-v2

policy:
  hit_distance_threshold: 0.18
  sage_ghost_capacity: 4096
  sage_decay_half_life_requests: 10000

quality:
  good_hit_distance_thresholds: [0.1, 0.2, 0.3]
```

The numerical thresholds are experiment parameters, not universal constants. Tune the prompt hit threshold on a validation split and freeze it before test comparison. Likewise, define answer-quality thresholds from a held-out sample or human-judged calibration set.

## Run

```bash
PYTHONPATH=. python -m experiments.run \
  --config experiments/configs/oasst1_default.yaml
```

## Raw output schema

Each compressed raw CSV includes:

- request and pair identifiers;
- warm-up/measured flag;
- policy and capacity;
- hit/miss;
- matched-prompt distance;
- returned-vs-reference response distance;
- simulated backend latency;
- measured policy overhead;
- total simulated latency;
- SAGE admission decision and net delta where applicable.

## Suggested paper-quality protocol

1. Choose train/validation/test partitions without leaking identical message trees across splits.
2. Calibrate semantic-hit and answer-quality thresholds only on validation data.
3. Run at least five trace seeds.
4. Report mean and confidence interval for hit rate, quality-adjusted hit rate, bad-hit rate, latency, and overhead.
5. Sweep cache capacity and ghost capacity.
6. Include SAGE ablations: no decay, hits-only ghost observations, and overlap-unaware scoring.
7. Inspect a stratified sample of good and bad hits manually or with a separate judged benchmark.
