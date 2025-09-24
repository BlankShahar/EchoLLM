from abc import abstractmethod, ABC


class ICache(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size

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
