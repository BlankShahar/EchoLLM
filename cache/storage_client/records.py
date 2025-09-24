import hashlib
from abc import abstractmethod, ABC

from pydantic import BaseModel


class Record(BaseModel, ABC):
    @abstractmethod
    @property
    def key(self) -> str:
        raise NotImplementedError


class ResponseRecord(Record):
    request_key: str
    response: str

    @property
    def key(self) -> str:
        return hashlib.md5(self.response.encode()).hexdigest()


class EmbeddedRequestRecord(Record):
    vector: list[float]

    @property
    def key(self) -> str:
        return hashlib.md5(str(self.vector).encode()).hexdigest()