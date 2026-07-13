from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class EchoLLMResponse(BaseModel):
    """Response plus the measurements produced by one EchoLLM request."""

    model_config = ConfigDict(frozen=True)

    response: str
    cache_hit: bool
    llm_latency: float = Field(ge=0.0)
    cache_lookup_latency: float = Field(ge=0.0)
    cache_admission_latency: float = Field(ge=0.0)
    cache_metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def cache_latency(self) -> float:
        return self.cache_lookup_latency + self.cache_admission_latency
