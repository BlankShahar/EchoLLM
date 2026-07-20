from pathlib import Path

import pytest

from _experiments.recorded_llm import RecordedLLM, RecordedLLMStore, RecordedResponse


def test_recorded_llm_replays_response_and_real_measured_latency(tmp_path: Path) -> None:
    path = tmp_path / "responses.sqlite3"
    store = RecordedLLMStore(path, writable=True)
    store.validate_or_initialize(model="model", options={"num_predict": 32})
    store.put(RecordedResponse(prompt="hello", response="world", latency_ms=12.5))
    store.close()

    backend = RecordedLLM(path)
    response = backend.ask("hello")
    backend.close()

    assert response.response == "world"
    assert response.latency == 12.5


def test_recorded_store_rejects_different_generation_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "responses.sqlite3"
    store = RecordedLLMStore(path, writable=True)
    store.validate_or_initialize(model="model", options={"num_predict": 32})
    with pytest.raises(ValueError, match="metadata"):
        store.validate_or_initialize(model="other", options={"num_predict": 32})
    store.close()
