from functools import lru_cache

from text_similarity import vector_utils
from ..ranking_distance_method import RankingDistanceMethod
from ...storage_client.faiss_client import FaissClient, FaissDistanceMethod
from ...storage_client.records import EmbeddedRequestRecord


class RequestsDB:
    _RANKING_METHODS_MAP = {
        RankingDistanceMethod.EUCLIDEAN: vector_utils.euclidean_distance,
        RankingDistanceMethod.MANHATTAN: vector_utils.manhattan_distance,
        RankingDistanceMethod.COSINE: vector_utils.cosine_distance,
    }

    def __init__(
            self,
            ranking_distance_method=RankingDistanceMethod.EUCLIDEAN,
            db_distance_method=FaissDistanceMethod.L2
    ):
        """
        :param db_distance_method: The distance method to use for handling the inner vector DB of the embedded requests.
            Defaults to DistanceMethod.L2, which is an Euclidean distance.
        :param ranking_distance_method: The distance method to use for picking the most similar request.
            The inner DB returns K most similar requests, and out of those K we pick the most similar based on this distance method.
        """
        self._faiss_client = FaissClient(db_distance_method)
        self._ranking_distance_method = ranking_distance_method

    @lru_cache
    def most_similar_request(self, embedded_request: list[float], k=100) -> tuple[EmbeddedRequestRecord, float]:
        """
        Returns the most similar (embedded, i.e. vectorized) question in the DB which were previously asked.
        """
        candidates = self._faiss_client.fetch_nearest_k(embedded_request, k)
        distance = self._RANKING_METHODS_MAP[self._ranking_distance_method]
        best_candidate, best_distance = min(
            [(candidate, distance(embedded_request, candidate.vector)) for candidate in candidates],
            key=lambda item: item[1],
        )
        return EmbeddedRequestRecord(key=best_candidate.key, vector=best_candidate.vector), best_distance

    def save(self, request: EmbeddedRequestRecord) -> str:
        key = self._faiss_client.save(request.vector, request.key)
        assert request.key == key
        return key

    def remove(self, key: str) -> bool:
        return self._faiss_client.remove(key)

    def size(self) -> int:
        """Retruns the amount of records in the DB."""
        return self._faiss_client.size()
