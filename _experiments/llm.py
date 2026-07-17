from collections.abc import Iterator, Mapping

from llm import ILLM, LLMResponse, LLMResponseChunk, Ollama

from .config import LLMConfig, LLMProvider


def build_llm(config: LLMConfig) -> ILLM:
    if config.provider == LLMProvider.OLLAMA:
        return Ollama(model=config.model, host=config.host, options=config.options)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


class ReferenceLLM(ILLM):
    """Deterministic backend for offline tests and synthetic _experiments."""

    def __init__(self, responses: Mapping[str, str], latency: float = 0.0):
        self._responses = dict(responses)
        self._latency = latency

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        return LLMResponse(response=self._responses[prompt], latency=self._latency)

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        yield LLMResponseChunk(
            response_chunk=self._responses[prompt],
            chunk_number=1,
            delay=self._latency,
        )
