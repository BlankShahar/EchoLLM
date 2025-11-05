import time
from typing import Any

from openai import OpenAI
from openai.types import ChatModel

from .illm import ILLM, LLMResponse


class ChatGPTResponse(LLMResponse):
    prompt_tokens: int
    response_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class ChatGPT(ILLM):
    def __init__(
            self,
            model: ChatModel,
            api_key: str,
            base_url: str,
            options: dict[str, Any] | None = None,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._options = options or {}

    def ask(self, prompt: str) -> ChatGPTResponse:
        start_time = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **self._options,
        )
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        return ChatGPTResponse(
            response=response.choices[0].message.content,
            time=elapsed_ms,
            prompt_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
        )
