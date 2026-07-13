from collections.abc import Iterator

from _experiments.llm import MemoizedLLM
from llm import ILLM, LLMResponse, LLMResponseChunk


class CountingLLM(ILLM):
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(response=f"generated:{prompt}", latency=12.5)

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        yield LLMResponseChunk(
            response_chunk=f"generated:{prompt}",
            chunk_number=1,
            delay=12.5,
        )


def test_memoized_llm_reuses_framework_response_and_latency() -> None:
    backend = CountingLLM()
    llm = MemoizedLLM(backend)

    llm.prime(["one", "one", "two"])
    replay = llm.ask("one")

    assert backend.calls == 2
    assert replay.response == "generated:one"
    assert replay.latency == 12.5
