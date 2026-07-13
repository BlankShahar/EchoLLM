# Feature-branch file map

## Modified EchoLLM framework files

- `cache/icache.py` — adds backward-compatible `lookup()`.
- `echollm/echollm.py` — consumes `CacheLookup`, forwards miss context, and exposes request measurements.
- `llm/ollama_llm.py` — accepts configured model names through the existing `ILLM` implementation.

## New framework files

- `cache/models.py`
- `echollm/models.py`
- `cache/sage_similarity_cache.py`
- `cache/sage/__init__.py`
- `cache/sage/cache.py`
- `cache/sage/config.py`
- `cache/sage/distance.py`
- `cache/sage/ghost_window.py`
- `cache/sage/models.py`
- `cache/sage/scorer.py`
- `cache/sage/storage.py`

## Experiment project

- `experiments/config.py`
- `experiments/oasst1.py`
- `experiments/embeddings.py`
- `experiments/trace.py`
- `experiments/baselines.py`
- `experiments/llm.py`
- `experiments/metrics.py`
- `experiments/resources.py`
- `experiments/runner.py`
- `experiments/plotting.py`
- `experiments/run.py`
- `experiments/plot.py`
- `experiments/smoke.py`
- `experiments/configs/oasst1_default.yaml`
- `experiments/slurm/run_oasst1.sbatch`

## Validation

- `tests/`
- `results/smoke/`

When applying this feature over a complete EchoLLM checkout, retain upstream modules not listed here. The included `llm/`, `text_similarity/`, and package `__init__.py` files are the minimal compatible core needed to run this artifact independently.
