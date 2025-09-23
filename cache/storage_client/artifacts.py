import hashlib
from abc import abstractmethod, ABC

from pydantic import BaseModel


class Artifact(BaseModel, ABC):
    @abstractmethod
    @property
    def key(self) -> str:
        raise NotImplementedError


class ResponseRecord(Artifact):
    request_key: str
    response: str

    @property
    def key(self) -> str:
        return hashlib.md5(self.response.encode()).hexdigest()


class EmbeddedRequestRecord(Artifact):
    vector: list[float]

    @property
    def key(self) -> str:
        return hashlib.md5(str(self.vector).encode()).hexdigest()