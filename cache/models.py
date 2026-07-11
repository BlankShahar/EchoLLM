from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CacheLookup(BaseModel):
    """Single-pass cache lookup result.

    ``context`` carries policy-specific state from lookup to miss completion so an
    expensive embedding or nearest-neighbour search does not need to be repeated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hit: bool
    response: Any | None = None
    context: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
