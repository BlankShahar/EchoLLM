from collections.abc import Iterable, Iterator, Mapping
from threading import RLock

from tqdm import tqdm

from llm import ILLM, LLMResponse, LLMResponseChunk, Ollama

from .config import LLMConfig, LLMProvider


def build_llm(config: LLMConfig) -> ILLM:
    if config.provider == LLMProvider.OLLAMA:
        return Ollama(model=config.model, host=config.host, options=config.options)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


class MemoizedLLM(ILLM):
    """Calls the backend once per prompt and replays the same LLMResponse."""

    def __init__(self, backend: ILLM):
        self._backend = backend
        self._responses: dict[str, LLMResponse] = {}
        self._lock = RLock()

    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        with self._lock:
            cached = self._responses.get(prompt)
            if cached is not None:
                return cached
            response = self._backend.ask(prompt, **kwargs)
            self._responses[prompt] = response
            return response

    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        return self._backend.stream_ask(prompt, **kwargs)

    def prime(self, prompts: Iterable[str]) -> None:
        unique_prompts = list(dict.fromkeys(prompts))
        for prompt in tqdm(unique_prompts, desc="Generating LLM responses", unit="prompt"):
            self.ask(prompt)

    @property
    def responses(self) -> dict[str, LLMResponse]:
        with self._lock:
            return dict(self._responses)


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
