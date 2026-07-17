import numpy as np

from _experiments.config import TraceConfig, TraceMode
from _experiments.models import PromptResponsePair
from _experiments.trace import build_trace


def test_chronological_trace_uses_every_pair_once_by_default() -> None:
    pairs = [
        _pair(0, "middle", "2023-02-02T00:00:00+00:00"),
        _pair(1, "newest", "2023-02-03T00:00:00+00:00"),
        _pair(2, "oldest", "2023-02-01T00:00:00+00:00"),
    ]
    trace = build_trace(
        pairs,
        np.eye(3, dtype=np.float32),
        TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
    )

    assert [request.prompt for request in trace] == ["oldest", "middle", "newest"]
    assert len(trace) == len(pairs)


def test_trace_preserves_dataset_source_model() -> None:
    pair = _pair(0, "prompt", "2023-02-01T00:00:00+00:00").model_copy(
        update={"source_model": "gpt-4-0314"}
    )

    trace = build_trace(
        [pair],
        np.ones((1, 1), dtype=np.float32),
        TraceConfig(mode=TraceMode.CHRONOLOGICAL, request_count=None),
    )

    assert trace[0].source_model == "gpt-4-0314"


def _pair(index: int, prompt: str, created_at: str) -> PromptResponsePair:
    return PromptResponsePair(
        pair_index=index,
        prompt_id=f"p-{index}",
        response_id=f"r-{index}",
        prompt=prompt,
        reference_response=f"answer-{index}",
        source_index=index,
        created_at=created_at,
    )
