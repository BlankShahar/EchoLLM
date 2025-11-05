import time
from enum import StrEnum
from typing import Any

import ollama

from .illm import ILLM, LLMResponse


class OllamaModel(StrEnum):
    QWEN3_4B = "qwen3:4b"
    QWEN3_8B = "qwen3:8b"
    QWEN3_30B = "qwen3:30b"
    QWEN3_CODER_480B = "qwen3-coder:480b"

    LLAMA3_1_8B = "llama3.1:8b"

    KIMI_K2_1T = "kimi-k2:1t"

    GPT_OSS_20B = "gpt-oss:20b"
    GPT_OSS_120B = "gpt-oss:120b"

    GEMMA3_1B = "gemma3:1b"
    GEMMA3_4B = "gemma3:4b"
    GEMMA3_12B = "gemma3:12b"
    GEMMA3_27B = "gemma3:27b"

    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    DEEPSEEK_V3_1_671B = "deepseek-v3.1:671b"


class OllamaResponse(LLMResponse):
    pass


class Ollama(ILLM):
    def __init__(self, model: OllamaModel, host: str, options: dict[str, Any] | None = None):
        self._client = ollama.Client(host=host)
        self._client.pull(model)
        self._model = model
        self._options = options or {}

    def ask(self, prompt: str) -> OllamaResponse:
        start_time = time.perf_counter()
        result = self._client.generate(
            model=self._model,
            prompt=prompt,
            options=self._options,
            stream=False,
        )
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return OllamaResponse(response=result.response, time=elapsed_ms)
