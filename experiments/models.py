from pydantic import BaseModel, ConfigDict, Field


class PromptResponsePair(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_index: int = Field(ge=0)
    prompt_id: str
    response_id: str
    message_tree_id: str | None = None
    prompt: str
    reference_response: str
    source_index: int = Field(ge=0)


class TraceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    prompt_id: str
    response_id: str
    prompt: str
    reference_response: str
