from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel, ABC):
    response: str
    latency: float = Field(description="Time taken for LLM to respond with the full answer, in milliseconds")


class StreamedLLMResponse(LLMResponse):
    delay: float = Field(description="Time taken for LLM to respond with the first token, in milliseconds")


class ILLM(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def stream_ask(self, prompt: str) -> StreamedLLMResponse:
        raise NotImplementedError
