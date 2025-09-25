import time
from enum import StrEnum

import ollama

from .illm import ILLM, LLMResponse


class OllamaModel(StrEnum):
    LLAMA3 = "llama3"
    LLAMA3_1 = "llama3.1"
    MISTRAL = "mistral"
    QWEN2 = "qwen2"
    PHI3 = "phi3"


class Ollama(ILLM):
    def __init__(self, model: OllamaModel, host: str | None, options: dict | None = None):
        self._model = model
        self._client = ollama.Client(host=host)
        self._options = options or {}

    def ask(self, prompt: str) -> LLMResponse:
        start_time = time.perf_counter()
        result = self._client.generate(
            model=self._model,
            prompt=prompt,
            options=self._options,
            stream=False,
        )
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return LLMResponse(response=result.response, time=elapsed_ms)
