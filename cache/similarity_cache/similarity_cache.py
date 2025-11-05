import hashlib
from abc import ABC
from typing import Callable

from cache import ICache
from .db_handlers import RequestsDB, ResponsesDB
from .ranking_distance_method import RankingDistanceMethod
from ..storage_client.faiss_client import FaissDistanceMethod


class SimilarityCache(ICache, ABC):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]],
            policy_name: str
    ):
        super().__init__(max_size, policy_name)
        self._hit_distance_threshold = hit_distance_threshold
        self._candidates_number = candidates_number
        self._requests_db = RequestsDB(ranking_distance_method, db_distance_method)
        self._responses_db = ResponsesDB()
        self._embedder = prompt_embedder

    def is_hit(self, prompt: str) -> bool:
        most_similar_request = self._requests_db.most_similar_request(
            self._embedder(prompt),
            self._candidates_number
        )
        if most_similar_request is None:
            return False
        _, distance = most_similar_request
        return distance <= self._hit_distance_threshold

    def on_hit(self, prompt: str, **kwargs) -> str:
        hit_request, _ = self._requests_db.most_similar_request(
            self._embedder(prompt),
            self._candidates_number
        )
        response = self._responses_db.fetch_by_request(hit_request.key)
        if response is None:
            raise KeyError(f'Response with request_key=`{hit_request.key}` was not found!')
        return response.response

    def current_size(self) -> int:
        return self._responses_db.size()

    @staticmethod
    def _generate_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _generate_int_key(text: str) -> int:
        return int(hashlib.md5(text.encode()).digest(), 16)
