from abc import ABC

from pydantic import BaseModel


class IRecord(BaseModel, ABC):
    key: str


class ResponseRecord(IRecord):
    request_key: str
    response: str


class EmbeddedRequestRecord(IRecord):
    vector: list[float]
