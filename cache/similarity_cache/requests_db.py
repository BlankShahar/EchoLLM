from enum import Enum, auto

from text_similarity import vector_utils
from ..storage_client.faiss_client import FaissClient, DistanceMethod
from ..storage_client.records import EmbeddedRequestRecord


class RankingDistanceMethod(Enum):
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    COSINE = auto()


class RequestsDB:
    _RANKING_METHODS_MAP = {
        RankingDistanceMethod.EUCLIDEAN: vector_utils.euclidean_distance,
        RankingDistanceMethod.MANHATTAN: vector_utils.manhattan_distance,
        RankingDistanceMethod.COSINE: vector_utils.cosine_distance,
    }

    def __init__(
            self,
            ranking_distance_method=RankingDistanceMethod.EUCLIDEAN,
            db_distance_method=DistanceMethod.L2
    ):
        """
        :param db_distance_method: The distance method to use for handling the inner vector DB of the embedded requests.
            Defaults to DistanceMethod.L2, which is an Euclidean distance.
        :param ranking_distance_method: The distance method to use for picking the most similar request.
            The inner DB returns K most similar requests, and out of those K we pick the most similar based on this distance method.
        """
        self.faiss_client = FaissClient(db_distance_method)
        self._ranking_distance_method = ranking_distance_method

    def most_similar_request(self, request: EmbeddedRequestRecord, k=100) -> EmbeddedRequestRecord:
        """
        Returns the most similar question in the DB which were previously asked.
        """
        candidates = self.faiss_client.fetch_nearest_k(request.vector, k)
        distance = self._RANKING_METHODS_MAP[self._ranking_distance_method]
        best_vector = min(candidates, key=lambda vector: distance(request.vector, vector))
        return EmbeddedRequestRecord(vector=best_vector)

    def save(self, request: EmbeddedRequestRecord) -> str:
        key = self.faiss_client.save(request.vector)
        assert request.key == key
        return key

    def remove(self, key: str) -> bool:
        return self.faiss_client.remove(key)
