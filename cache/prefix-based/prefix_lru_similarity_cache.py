import math
from typing import Callable

from pydantic import BaseModel

from cache.lru_similarity_cache import LRUSimilarityCache
from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.storage_client.faiss_client import FaissDistanceMethod
from cache.storage_client.records import EmbeddedRequestRecord, ResponseRecord


class DelayStats(BaseModel):
    mean: float  # E[x]
    m2: float  # E[x^2]
    observations: int = 1

    @property
    def std(self) -> float:
        # Var(x) = E[x^2] - E[x]^2 -> Std(x) = sqrt(Var(x))
        # max is used to handle floating point drift
        return math.sqrt(max(self.m2 - self.mean ** 2, 0))


class ItemStats(BaseModel):
    key: str
    delay: DelayStats


class MissingKwargError(TypeError):
    def __init__(self, arg):
        super().__init__(f'Missing keyword argument {arg}')


class PrefixLRUSimilarityCache(LRUSimilarityCache):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]],
            delay_ewma_alpha: float = 0.2,
            bandwidth: float = 1000,  # Mbps
            prefix_alpha: float = 2,
    ):
        super().__init__(
            max_size,
            hit_distance_threshold,
            candidates_number,
            ranking_distance_method,
            db_distance_method,
            prompt_embedder,
        )
        if not 0 < delay_ewma_alpha <= 1:
            raise ValueError('delay_ewma_alpha must be between 0 and 1')

        self.policy_name = 'Prefix-Based Similarity LRU'
        self.delay_ewma_alpha = delay_ewma_alpha
        self.bandwidth = bandwidth
        self.prefix_alpha = prefix_alpha
        self.itemwise_stats: dict[str, ItemStats] = {}

    def _update_item_stats(self, prompt_key: str, llm_delay: float):
        if prompt_key in self.itemwise_stats:
            item_stats = self.itemwise_stats[prompt_key]
            item_stats.delay.observations += 1

            alpha = self.delay_ewma_alpha
            old_mean = item_stats.delay.mean
            old_m2 = item_stats.delay.m2
            item_stats.delay.mean = (1 - alpha) * old_mean + alpha * llm_delay
            item_stats.delay.m2 = (1 - alpha) * old_m2 + alpha * (llm_delay ** 2)

        else:
            self.itemwise_stats[prompt_key] = ItemStats(
                key=prompt_key,
                delay=DelayStats(
                    mean=llm_delay,
                    m2=llm_delay ** 2,  # m2 as x^2 -> std starts at 0
                ),
            )

    def on_miss(self, prompt: str, llm_response: str, **kwargs):
        if 'llm_delay' not in kwargs:
            raise MissingKwargError('llm_delay')

        prompt_key = self._generate_key(prompt)
        self._update_item_stats(prompt_key, kwargs['llm_delay'])

        self._lru_cache[prompt_key] = True

        # if the last insert caused an eviction due to reaching maximum capacity
        if len(self._lru_cache) == self._max_size and self._lru_cache.last_evicted is not None:
            prompt_key, _ = self._lru_cache.last_evicted
            self._requests_db.remove(prompt_key)
            self._responses_db.remove_by_request(prompt_key)

        self._requests_db.save(
            EmbeddedRequestRecord(key=prompt_key, vector=self._embedder(prompt))
        )
        # TODO: cut llm_response to its prefix
        item_stats = self.itemwise_stats[prompt_key]
        prefix_size = self.bandwidth * (item_stats.delay.mean + self.prefix_alpha * item_stats.delay.std)
        prefix_llm_response = llm_response[:prefix_size]
        response_key = self._generate_key(llm_response)
        self._responses_db.save(
            ResponseRecord(key=response_key, request_key=prompt_key, response=prefix_llm_response)
        )
