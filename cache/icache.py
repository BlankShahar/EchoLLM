from abc import abstractmethod, ABC

from llm.llm import LLM


class ICache(ABC):
    def __init__(self, max_size: int, llm: LLM):
        self._max_size = max_size
        self._llm = llm

    @abstractmethod
    def on_request(self, item: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_hit(self, item_key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_hit(self, item_key: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_miss(self, item: str, **kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def current_size(self) -> int:
        raise NotImplementedError
