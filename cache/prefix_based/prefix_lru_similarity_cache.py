from typing import Callable, Any

from cachetools import LRUCache

from cache.prefix_based.errors import MissingKwargError
from cache.prefix_based.prefix_similarity_cache import PrefixSimilarityCache
from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.storage_client.faiss_client import FaissDistanceMethod
from cache.storage_client.records import EmbeddedRequestRecord, ResponseRecord


class HookedLRUCache(LRUCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_evicted = None

    def popitem(self) -> tuple[Any, Any]:
        k, v = super().popitem()  # this is called when the cache evicts
        self.last_evicted = (k, v)
        return k, v


class PrefixLRUSimilarityCache(PrefixSimilarityCache):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]],
            bandwidth: float = 1000,  # Mbps
            delay_ewma_smoothing_factor: float = 0.2,
            prefix_size_confidence_factor: float = 2,
    ):
        super().__init__(
            max_size,
            hit_distance_threshold,
            candidates_number,
            ranking_distance_method,
            db_distance_method,
            prompt_embedder,
            'Prefix-Based LRU Similarity Cache',
            bandwidth,
            delay_ewma_smoothing_factor,
            prefix_size_confidence_factor,
        )
        self._lru_cache = HookedLRUCache(max_size)

    def on_miss(self, prompt: str, llm_response: str, **kwargs):
        if 'llm_delay' not in kwargs:
            raise MissingKwargError('llm_delay')

        prompt_key = self._generate_key(prompt)
        self.update_item_stats(prompt_key, kwargs['llm_delay'])

        self._lru_cache[prompt_key] = True

        # if the last insert caused an eviction due to reaching maximum capacity
        if len(self._lru_cache) == self._max_size and self._lru_cache.last_evicted is not None:
            prompt_key, _ = self._lru_cache.last_evicted
            self._requests_db.remove(prompt_key)
            self._responses_db.remove_by_request(prompt_key)

        self._requests_db.save(
            EmbeddedRequestRecord(key=prompt_key, vector=self._embedder(prompt))
        )
        item_stats = self.itemwise_stats[prompt_key]
        prefix_size = self.bandwidth * (
                item_stats.delay.mean + self.prefix_size_confidence_factor * item_stats.delay.std)
        prefix_llm_response = llm_response[:prefix_size]
        response_key = self._generate_key(llm_response)
        self._responses_db.save(
            ResponseRecord(key=response_key, request_key=prompt_key, response=prefix_llm_response)
        )
