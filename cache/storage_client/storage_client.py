from abc import abstractmethod, ABC

from .artifacts import Artifact


class StorageClient(ABC):
    @abstractmethod
    def fetch(self, key: str) -> Artifact:
        raise NotImplementedError

    @abstractmethod
    def save(self, artifact: Artifact) -> str:
        raise NotImplementedError
