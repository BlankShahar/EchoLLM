from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel, ABC):
    response: str
    time: float = Field(description="Time taken for LLM to respond, in milliseconds")


class ILLM(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
