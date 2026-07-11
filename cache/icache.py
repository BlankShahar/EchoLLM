from abc import ABC, abstractmethod
from typing import Any

from .models import CacheLookup


class ICache(ABC):
    def __init__(self, max_size: int, policy_name: str):
        if max_size <= 0:
            raise ValueError("max_size must be greater than zero")
        self._max_size = max_size
        self.policy_name = policy_name

    def lookup(self, request: Any) -> CacheLookup:
        """Backward-compatible single-pass lookup.

        Existing policies continue to work through ``is_hit``/``on_hit``. New
        policies should override this method to avoid duplicate work.
        """
        if self.is_hit(request):
            return CacheLookup(hit=True, response=self.on_hit(request))
        return CacheLookup(hit=False)

    @abstractmethod
    def is_hit(self, request: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_hit(self, request: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def on_miss(self, request: Any, response: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_size(self) -> int:
        raise NotImplementedError
