from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, Field


class LLMResponse(BaseModel, ABC):
    response: str
    latency: float = Field(description="Time taken for LLM to respond with the full answer, in milliseconds")


class LLMResponseChunk(BaseModel):
    response_chunk: str
    chunk_number: int = Field(ge=1)
    delay: float = Field(description="Time taken for LLM to respond with this chunk, in milliseconds")

    @property
    def is_first(self) -> bool:
        return self.chunk_number == 1


class ILLM(ABC):
    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def stream_ask(self, prompt: str, **kwargs) -> Iterator[LLMResponseChunk]:
        raise NotImplementedError
