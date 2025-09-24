import hashlib
from abc import abstractmethod, ABC

from pydantic import BaseModel, computed_field


class Record(BaseModel, ABC):
    @computed_field
    @property
    @abstractmethod
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