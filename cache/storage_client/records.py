import hashlib
from abc import abstractmethod, ABC

from pydantic import BaseModel, computed_field


class IRecord(BaseModel, ABC):
    @computed_field
    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError


class ResponseRecord(IRecord):
    request_key: str
    response: str

    @property
    def key(self) -> str:
        return hashlib.md5(self.response.encode()).hexdigest()


class EmbeddedRequestRecord(IRecord):
    vector: list[float]

    @property
    def key(self) -> str:
        return hashlib.md5(str(self.vector).encode()).hexdigest()