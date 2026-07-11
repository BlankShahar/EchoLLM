from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    response: str
    latency: float = Field(ge=0, description="Full-response latency in milliseconds")


class LLMResponseChunk(BaseModel):
    response_chunk: str
    chunk_number: int = Field(ge=1)
    delay: float = Field(ge=0, description="Chunk latency in milliseconds")

    @property
    def is_first(self) -> bool:
        return self.chunk_number == 1


class ILLM(ABC):
    @abstractmethod
    def ask(self, prompt: str, **kwargs: object) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def stream_ask(self, prompt: str, **kwargs: object) -> Iterator[LLMResponseChunk]:
        raise NotImplementedError
