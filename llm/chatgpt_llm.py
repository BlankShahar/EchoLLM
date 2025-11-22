import time
from typing import Any

from openai import OpenAI, Stream, BaseModel
from openai.types import ChatModel
from openai.types.chat import ChatCompletionChunk

from .illm import ILLM, LLMResponse, StreamedLLMResponse


class ResponseTokens(BaseModel):
    prompt_tokens: int
    response_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class ChatGPTResponse(LLMResponse, ResponseTokens):
    pass


class StreamedChatGPTResponse(StreamedLLMResponse, ResponseTokens):
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

    def stream_ask(self, prompt: str) -> StreamedChatGPTResponse:
        start_time = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **self._options,
        )
        first_token_time = None
        prompt_tokens, response_tokens = 0, 0
        full_response = ''

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                response_tokens = chunk.usage.completion_tokens
            full_response += chunk.choices[0].delta.content
        end_time = time.perf_counter()

        return StreamedChatGPTResponse(
            response=full_response,
            latency=(end_time - start_time) * 1000,
            delay=(first_token_time - start_time) * 1000,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )
