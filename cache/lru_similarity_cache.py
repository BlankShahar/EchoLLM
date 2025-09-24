from typing import Callable, Any

from cachetools import LRUCache

from .similarity_cache import SimilarityCache
from .similarity_cache.ranking_distance_method import RankingDistanceMethod
from .storage_client.faiss_client import FaissDistanceMethod
from .storage_client.records import EmbeddedRequestRecord, ResponseRecord


class HookedLRUCache(LRUCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_evicted = None

    def popitem(self) -> tuple[Any, Any]:
        k, v = super().popitem()  # this is called when the cache evicts
        self.last_evicted = (k, v)
        return k, v


class LRUSimilarityCache(SimilarityCache):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]]
    ):
        super().__init__(
            max_size,
            hit_distance_threshold,
            candidates_number,
            ranking_distance_method,
            db_distance_method,
            prompt_embedder
        )
        self._lru_cache = HookedLRUCache(max_size)

    def on_miss(self, prompt: str, **kwargs) -> None:
        prompt_key = self._generate_key(
            prompt)  # TODO: fix the generate key bamboozle into a single one source of truth
        self._lru_cache[prompt_key] = True

        # if the last insert generated an eviction
        if len(self._lru_cache) == self._max_size and self._lru_cache.last_evicted is not None:
            prompt_key, _ = self._lru_cache.last_evicted
            self._requests_db.remove(prompt_key)
            self._responses_db.remove_by_request(prompt_key)

        self._requests_db.save(EmbeddedRequestRecord(vector=self._embedder(prompt)))
        llm_response = self._llm.ask(prompt)
        self._responses_db.save(ResponseRecord(request_key=prompt_key, response=llm_response.response))
