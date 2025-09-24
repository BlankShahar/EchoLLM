import hashlib
from abc import ABC
from typing import Callable

from cache import ICache
from .db_handlers import RequestsDB, ResponsesDB
from .ranking_distance_method import RankingDistanceMethod
from ..storage_client.faiss_client import FaissDistanceMethod
from ..storage_client.records import EmbeddedRequestRecord


class SimilarityCache(ICache, ABC):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]]
    ):
        super().__init__(max_size)
        self._hit_distance_threshold = hit_distance_threshold
        self._candidates_number = candidates_number
        self._requests_db = RequestsDB(ranking_distance_method, db_distance_method)
        self._responses_db = ResponsesDB()
        self._embedder = prompt_embedder

    def on_request(self, prompt: str, **kwargs) -> str | None:
        prompt_key = self._generate_key(prompt)
        if self.is_hit(prompt_key):
            return self.on_hit(prompt_key)
        self.on_miss(prompt)
        return None

    def is_hit(self, prompt: str) -> bool:
        _, distance = self._requests_db.most_similar_request(
            EmbeddedRequestRecord(vector=self._embedder(prompt)),
            self._candidates_number
        )
        return distance <= self._hit_distance_threshold

    def on_hit(self, prompt_key: str, **kwargs) -> str:
        response = self._responses_db.fetch(prompt_key)
        if response is None:
            raise KeyError(prompt_key)
        return response.response

    def current_size(self) -> int:
        return self._responses_db.size()

    def _generate_key(self, prompt: str) -> str:
        embedded_prompt = self._embedder(prompt)
        return hashlib.md5(str(embedded_prompt).encode()).hexdigest()
