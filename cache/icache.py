from abc import abstractmethod, ABC
from typing import Any


class ICache(ABC):
    def __init__(self, max_size: int, policy_name: str):
        self._max_size = max_size
        self.policy_name = policy_name

    @abstractmethod
    def is_hit(self, request: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_hit(self, request: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def on_miss(self, request: Any, response: Any, **kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_size(self) -> int:
        raise NotImplementedError
