# EchoLLM SAGE feature branch

This package adds **SAGE — Semantic Admission and Gain-based Eviction** to EchoLLM and includes a reproducible OASST1 experiment harness.

SAGE treats a semantic cache as a set of overlapping coverage regions. It records a bounded ghost window of recent request embeddings and admits a missed prompt only when replacing a resident strictly increases recency-weighted semantic coverage.

## Included changes

- `cache/models.py`: Pydantic `CacheLookup` result.
- `cache/icache.py`: backward-compatible `lookup()` API.
- `echollm/echollm.py`: one-pass lookup and lookup-context propagation.
- `cache/sage/`: policy, exact distance engine, ghost window, vectorized scorer, persistence, Pydantic models/configuration.
- `cache/sage_similarity_cache.py`: convenient import compatible with EchoLLM's flat policy modules.
- `experiments/`: OASST1 loader, trace generation, exact semantic baselines, response-quality evaluation, raw CSV output, summaries, and plots.
- `tests/`: mathematical equivalence, policy behavior, persistence, framework flow, dataset selection, and metric tests.

The package is organized as a feature-branch overlay: copy its changed/new modules over an EchoLLM checkout, or use this self-contained core directly for policy and experiment work. Existing policies remain compatible because `ICache.lookup()` falls back to their original `is_hit()`/`on_hit()` methods.

## Install

Core policy:

```bash
python -m pip install -r requirements.txt
```

Full OASST1 experiments:

```bash
python -m pip install -r requirements-experiments.txt
```

Python 3.11+ is recommended.

## Minimal use

```python
from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod
from echollm import EchoLLM

cache = SAGESimilarityCache(
    max_size=1_000,
    hit_distance_threshold=0.18,
    prompt_embedder=my_embedding_function,
    ranking_distance_method=RankingDistanceMethod.COSINE,
    ghost_capacity=4_096,
    decay_half_life_requests=10_000,
    storage_path=".cache/sage.sqlite3",
    storage_namespace="production-v1",
)

echo = EchoLLM(cache=cache, llm=my_llm)
answer = echo.ask("How can I reset my password?")
```

`prompt_embedder` must accept a string and return one numeric vector. The response does not influence SAGE-HR's admission score; the policy objective is semantic hit coverage only.

## Run tests

```bash
PYTHONPATH=. pytest -q
```

## Run a deterministic smoke experiment

```bash
PYTHONPATH=. python -m experiments.smoke
```

This produces raw per-request files, `summary.csv`, `summary.json`, and plots under `results/smoke/`.

## Run OASST1

```bash
PYTHONPATH=. python -m experiments.run \
  --config experiments/configs/oasst1_default.yaml
```

Outputs are written to `results/oasst1_sage/` by default:

```text
results/oasst1_sage/
├── config.json
├── summary.csv
├── summary.json
├── raw/*.csv.gz
└── plots/*.png
```

Regenerate plots without rerunning the trace:

```bash
PYTHONPATH=. python -m experiments.plot \
  --results-dir results/oasst1_sage
```

## Main experimental metrics

- **Semantic hit rate**: all semantic cache hits divided by all measured requests.
- **Hit-only response cosine distance**: cosine distance between the returned cached response and the OASST1 reference response for that request.
- **Quality-adjusted hit rate at threshold `t`**: hits whose response distance is at most `t`, divided by **all** measured requests.
- **Bad-hit rate at threshold `t`**: hits whose distance exceeds `t`, divided by all measured requests.
- **Mean simulated latency**: cache overhead plus a configurable miss-latency model.
- **Mean policy overhead**: measured local lookup/admission runtime.

The quality-adjusted hit rate is the central anti-reward-hacking metric. Hit rate alone rewards overly permissive matching; hit-only response distance alone can reward a policy that serves only a few easy hits. Their joint metric rewards many hits only when those hits return a sufficiently similar answer.

Prompt embeddings and response-quality embeddings use separate configurable models by default. This avoids directly evaluating answer quality in the same vector space that drives cache matching.

## Important fairness choices

- Every policy uses the same exact resident scan, prompt embedding model, distance function, hit threshold, trace, cache capacity, and warm-up.
- Baselines are LRU, LFU, FIFO, and random replacement.
- Misses return the OASST1 reference response, so their response distance is zero in the optional end-to-end metric. Therefore, end-to-end response distance must not be interpreted without hit rate.
- Raw results record each hit, prompt distance, response distance, estimated LLM latency, policy overhead, and SAGE admission delta.
- `response_selection: top_rank` avoids ambiguity from evaluating the same prompt against several valid OASST1 answers.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and [`experiments/README.md`](experiments/README.md) for the detailed design and methodology.
