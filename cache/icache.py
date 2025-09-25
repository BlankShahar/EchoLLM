from abc import abstractmethod, ABC
from typing import Any

from llm.illm import ILLM


class ICache(ABC):
    def __init__(self, max_size: int, llm: ILLM):
        self._max_size = max_size
        self._llm = llm

    @abstractmethod
    def on_request(self, request: Any, **kwargs) -> Any | None:
        """Returns cached item in case of hit, None otherwise."""
        raise NotImplementedError

    @abstractmethod
    def is_hit(self, request: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_hit(self, request: Any, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_miss(self, request: Any, **kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_size(self) -> int:
        raise NotImplementedError
