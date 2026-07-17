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
from _experiments.runner import ExperimentRunner, format_duration
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


def test_run_grid_has_stable_capacity_then_policy_order() -> None:
    runner = _runner(
        pairs=[_pair(0), _pair(1)],
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["LRU", "SAGE"],
            cache_sizes=[0, 1],
            include_unbounded_cache=True,
        ),
    )

    assert runner.run_grid() == [
        ("LRU", 0, "no_cache", 0),
        ("SAGE", 0, "no_cache", 1),
        ("LRU", 1, "bounded", 2),
        ("SAGE", 1, "bounded", 3),
        ("LRU", 2, "unbounded", 4),
        ("SAGE", 2, "unbounded", 5),
    ]


def test_runner_passes_windowed_sage_configuration() -> None:
    runner = _runner(
        pairs=[_pair(0), _pair(1), _pair(2)],
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["SAGE"],
            cache_sizes=[3],
            include_unbounded_cache=False,
            sage_window_fraction=0.34,
            sage_soft_coverage=False,
            sage_soft_coverage_power=2.0,
            sage_ghost_capacity=7,
            sage_long_history_capacity=9,
            sage_long_sample_stride=3,
            sage_recent_evidence_weight=0.6,
            sage_long_decay_half_life_requests=100.0,
        ),
    )

    cache = runner._build_cache("SAGE", 3)

    assert cache is not None
    assert cache.config.window_size == 1
    assert cache.config.soft_coverage is False
    assert cache.config.soft_coverage_power == 2.0
    assert cache.config.recent_capacity == 7
    assert cache.config.long_capacity == 9
    assert cache.config.long_sample_stride == 3
    assert cache.config.recent_evidence_weight == 0.6
    assert cache.config.long_decay_half_life_requests == 100.0


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


def test_runner_calls_framework_llm_on_every_cache_miss(
    tmp_path: Path,
    capsys,
) -> None:
    backend = GeneratedLLM()
    pair = _pair(0).model_copy(update={"source_model": "source-model"})
    config = ExperimentConfig(
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=2),
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
    sage_raw = pd.read_csv(output / "raw" / "sage_cache_1.csv.gz")

    # Both no-cache runs call the backend twice for the repeated exact prompt;
    # each capacity-one run calls it once and serves the repetition from cache.
    assert backend.calls == 6
    assert len(summary) == 4
    assert len(raw) == 2
    assert not raw["hit"].any()
    assert raw.loc[0, "backend_latency_ms"] == 25.0
    assert raw.loc[0, "response_cosine_distance"] == 0.0
    assert raw.loc[0, "source_model"] == "source-model"
    assert bool(sage_raw.loc[0, "incoming_admitted"])
    assert "promoted" in sage_raw.columns
    assert not list((output / "raw").glob("*.partial"))
    captured = capsys.readouterr().out
    assert "LRU cache=0 (no_cache)" in captured
    assert "completed in 00:00:" in captured


def test_runner_executes_only_the_selected_array_grid_entry(tmp_path: Path) -> None:
    backend = GeneratedLLM()
    pair = _pair(0)
    config = ExperimentConfig(
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["LRU", "SAGE"],
            cache_sizes=[0],
            include_unbounded_cache=False,
        ),
        resources=ResourceConfig(enabled=False),
        output=OutputConfig(
            directory=tmp_path,
            run_name="selected-grid-entry",
            write_raw_results=False,
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

    output = runner.run(run_index=1)
    summary = pd.read_csv(output / "summary.csv")

    assert backend.calls == 1
    assert list(summary["policy"]) == ["SAGE"]


def test_failed_run_keeps_partial_raw_output(tmp_path: Path) -> None:
    pair = _pair(0)
    config = ExperimentConfig(
        trace=TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
        policy=PolicyConfig(
            policies=["LRU"],
            cache_sizes=[1],
            include_unbounded_cache=False,
        ),
        resources=ResourceConfig(enabled=False),
        output=OutputConfig(
            directory=tmp_path,
            run_name="partial-output",
            write_raw_results=True,
            generate_plots=False,
        ),
    )
    runner = ExperimentRunner(
        config,
        [pair],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    def fail_quality_evaluation(returned: str, reference: str) -> float:
        raise RuntimeError("quality evaluator failed")

    runner._response_distance = fail_quality_evaluation

    try:
        runner._run_one("LRU", 1, "bounded", raw_directory)
    except RuntimeError as error:
        assert str(error) == "quality evaluator failed"
    else:
        raise AssertionError("Expected the policy run to fail")

    assert (raw_directory / "lru_cache_1.csv.gz.partial").exists()
    assert not (raw_directory / "lru_cache_1.csv.gz").exists()


def test_format_duration_uses_fixed_width_hours_minutes_seconds() -> None:
    assert format_duration(0.4) == "00:00:00"
    assert format_duration(3661.4) == "01:01:01"


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
