from collections.abc import Iterator

from cache.sage import SAGESimilarityCache
from echollm import EchoLLM
from llm import ILLM, LLMResponse, LLMResponseChunk


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]


class DummyLLM(ILLM):
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(response="answer", latency=10.0)

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        yield LLMResponseChunk(response_chunk="answer", chunk_number=1, delay=10.0)


def test_echo_llm_performs_one_embedding_per_request() -> None:
    embedder = CountingEmbedder()
    llm = DummyLLM()
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
    )
    echo = EchoLLM(cache=cache, llm=llm)

    assert echo.ask("first") == "answer"
    assert embedder.calls == 1
    assert llm.calls == 1

    assert echo.ask("semantically identical") == "answer"
    assert embedder.calls == 2
    assert llm.calls == 1
