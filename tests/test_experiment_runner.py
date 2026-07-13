import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from _experiments.config import (
    ExperimentConfig,
    OutputConfig,
    PolicyConfig,
    ResourceConfig,
    TraceConfig,
    TraceMode,
)
from _experiments.models import PromptResponsePair
from _experiments.runner import ExperimentRunner
from llm import ILLM, LLMResponse, LLMResponseChunk


def test_capacity_runs_include_zero_and_resolved_unbounded() -> None:
    runner = _runner(
        pairs=[_pair(0), _pair(1), _pair(2)],
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["LRU"],
            cache_sizes=[0, 1],
            include_unbounded_cache=True,
        ),
    )

    assert runner._capacity_runs() == [
        (0, "no_cache"),
        (1, "bounded"),
        (3, "unbounded"),
    ]


def test_quality_evaluation_time_is_not_policy_overhead() -> None:
    runner = _runner(
        pairs=[_pair(0)],
        trace=TraceConfig(mode=TraceMode.DATASET_ORDER, request_count=2),
        policy=PolicyConfig(
            policies=["LRU"],
            cache_sizes=[1],
            include_unbounded_cache=False,
            hit_distance_threshold=0.01,
        ),
    )

    def slow_quality_evaluation(returned: str, reference: str) -> float:
        time.sleep(0.05)
        return 0.0

    runner._response_distance = slow_quality_evaluation
    summary = runner._run_one("LRU", 1, "bounded", Path("."))

    assert summary.hits == 1
    assert summary.mean_policy_overhead_ms < 25.0


def test_runner_uses_one_framework_llm_response_for_every_policy(
    tmp_path: Path,
) -> None:
    backend = GeneratedLLM()
    pair = _pair(0)
    config = ExperimentConfig(
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["LRU", "SAGE"],
            cache_sizes=[0, 1],
            include_unbounded_cache=False,
        ),
        resources=ResourceConfig(enabled=False),
        output=OutputConfig(
            directory=tmp_path,
            run_name="framework-llm",
            write_raw_results=True,
            generate_plots=False,
        ),
    )
    runner = ExperimentRunner(
        config,
        [pair],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        llm=backend,
        quality_provider=FixedQualityProvider(),
    )

    output = runner.run()
    summary = pd.read_csv(output / "summary.csv")
    raw = pd.read_csv(output / "raw" / "lru_cache_0.csv.gz")

    assert backend.calls == 1
    assert len(summary) == 4
    assert raw.loc[0, "backend_latency_ms"] == 25.0
    assert raw.loc[0, "response_cosine_distance"] == 0.0


def _runner(
    *,
    pairs: list[PromptResponsePair],
    trace: TraceConfig,
    policy: PolicyConfig,
) -> ExperimentRunner:
    vectors = np.eye(len(pairs), dtype=np.float32)
    config = ExperimentConfig(
        trace=trace,
        policy=policy,
        resources=ResourceConfig(enabled=False),
        output=OutputConfig(write_raw_results=False, generate_plots=False),
    )
    return ExperimentRunner(config, pairs, vectors, vectors)


def _pair(index: int) -> PromptResponsePair:
    return PromptResponsePair(
        pair_index=index,
        prompt_id=f"p-{index}",
        response_id=f"r-{index}",
        prompt=f"prompt-{index}",
        reference_response=f"response-{index}",
        source_index=index,
        created_at=f"2023-02-{index + 1:02d}T00:00:00+00:00",
    )


class GeneratedLLM(ILLM):
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(response=f"generated:{prompt}", latency=25.0)

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        yield LLMResponseChunk(
            response_chunk=f"generated:{prompt}",
            chunk_number=1,
            delay=25.0,
        )


class FixedQualityProvider:
    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.repeat(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            len(texts),
            axis=0,
        )
