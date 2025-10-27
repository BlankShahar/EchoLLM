import logging
from typing import Callable, Any

from cachetools import LFUCache

from .similarity_cache import SimilarityCache
from .similarity_cache.ranking_distance_method import RankingDistanceMethod
from .storage_client.faiss_client import FaissDistanceMethod
from .storage_client.records import EmbeddedRequestRecord, ResponseRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EchoLLM')


class HookedLFUCache(LFUCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_evicted = None

    def popitem(self) -> tuple[Any, Any]:
        k, v = super().popitem()  # this is called when the cache evicts
        self.last_evicted = (k, v)
        return k, v


class LFUSimilarityCache(SimilarityCache):
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
        self._lfu_cache = HookedLFUCache(max_size)

    def on_miss(self, prompt: str, llm_response: str, **kwargs):
        prompt_key = self._generate_key(prompt)
        self._lfu_cache[prompt_key] = True

        # if the last insert caused an eviction due to reaching maximum capacity
        if len(self._lfu_cache) == self._max_size and self._lfu_cache.last_evicted is not None:
            prompt_key, _ = self._lfu_cache.last_evicted
            self._requests_db.remove(prompt_key)
            self._responses_db.remove_by_request(prompt_key)

        self._requests_db.save(
            EmbeddedRequestRecord(key=prompt_key, vector=self._embedder(prompt))
        )
        response_key = self._generate_key(llm_response)
        self._responses_db.save(
            ResponseRecord(key=response_key, request_key=prompt_key, response=llm_response)
        )
