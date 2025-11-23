import time
from typing import Any, Iterator, Optional

from openai import OpenAI, BaseModel
from openai.types import ChatModel

from .illm import ILLM, LLMResponse, LLMResponseChunk


class ResponseTokens(BaseModel):
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class ChatGPTResponse(LLMResponse, ResponseTokens):
    pass


class ChatGPTResponseChunk(LLMResponseChunk, ResponseTokens):
    """
    Note: `prompt_tokens` and `response_tokens` will be set *only* for the last chunk.
            This is due to chatgpt interior implementation.
    """
    pass


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
            latency=elapsed_ms,
            prompt_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
        )

    def stream_ask(self, prompt: str) -> Iterator[ChatGPTResponseChunk]:
        start_time = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **self._options,
        )

        for i, chunk in enumerate(stream, start=1):
            prompt_tokens = chunk.usage.prompt_tokens if chunk.usage else None
            response_tokens = chunk.usage.completion_tokens if chunk.usage else None
            chunk_response = chunk.choices[0].delta.content
            current_time = time.perf_counter()

            yield ChatGPTResponseChunk(
                response_chunk=chunk_response,
                delay=(current_time - start_time) * 1000,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
