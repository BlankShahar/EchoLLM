import math
from abc import ABC
from typing import Callable

from pydantic import BaseModel

from cache.prefix_based.errors import MissingKwargError
from cache.similarity_cache import SimilarityCache
from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.storage_client.faiss_client import FaissDistanceMethod


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


class IPrefixSimilarityCache(SimilarityCache, ABC):
    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float,
            candidates_number: int,
            ranking_distance_method: RankingDistanceMethod,
            db_distance_method: FaissDistanceMethod,
            prompt_embedder: Callable[[str], list[float]],
            policy_name: str,
            bandwidth: float,
            delay_ewma_smoothing_factor: float = 0.2,
            prefix_size_confidence_factor: float = 2,
    ):
        if not 0 < delay_ewma_smoothing_factor <= 1:
            raise ValueError('delay_ewma_smoothing_factor must be between 0 and 1')

        super().__init__(
            max_size,
            hit_distance_threshold,
            candidates_number,
            ranking_distance_method,
            db_distance_method,
            prompt_embedder,
            policy_name
        )
        self.delay_ewma_smoothing_factor = delay_ewma_smoothing_factor
        self.bandwidth = bandwidth
        self.prefix_size_confidence_factor = prefix_size_confidence_factor
        self.itemwise_stats: dict[str, ItemStats] = {}

    def _update_delay_stats(self, prompt_key: str, llm_delay: float):
        if prompt_key in self.itemwise_stats:
            item_stats = self.itemwise_stats[prompt_key]
            item_stats.delay.observations += 1

            alpha = self.delay_ewma_smoothing_factor
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

    def update_item_stats(self, prompt_key: str, **kwargs):
        if 'llm_delay' not in kwargs:
            raise MissingKwargError('llm_delay')
        self._update_delay_stats(prompt_key, kwargs['llm_delay'])

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
        if not kwargs.get('retrieve_only'):
            self.update_item_stats(hit_request.key, **kwargs)
        response = self._responses_db.fetch_by_request(hit_request.key)
        if response is None:
            raise KeyError(f'Response with request_key=`{hit_request.key}` was not found!')
        return response.response
