import logging
from typing import Callable, Any

from adaptive_pipeline import AdaptivePipelineCache

from .similarity_cache import SimilarityCache
from .similarity_cache.ranking_distance_method import RankingDistanceMethod
from .storage_client.faiss_client import FaissDistanceMethod
from .storage_client.records import EmbeddedRequestRecord, ResponseRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EchoLLM')


class HookedAdaptivePipelineCache(AdaptivePipelineCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_evicted = None

    def popitem(self) -> tuple[Any, Any]:
        k, v = super().popitem()  # this is called when the cache evicts
        self.last_evicted = (k, v)
        return k, v


class MissingArgumentError(Exception):
    pass


class AdaptivePipelineSimilarityCache(SimilarityCache):
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
            prompt_embedder,
            'Similarity Adaptive-Pipeline'
        )
        self._ap_cache = HookedAdaptivePipelineCache(max_size)

    def on_miss(self, prompt: str, llm_response: str, **kwargs):
        """
        This function expects getting a kwarg argument named "llm_response_time", for the time the LLM took to answer.
            Otherwise, it will raise an exception.
        """
        llm_response_time = kwargs.get('llm_response_time')
        if llm_response_time is None:
            raise MissingArgumentError('Adaptive Pipeline policy requires "llm_response_time" argument!')

        prompt_key = self._generate_int_key(prompt)
        self._ap_cache[prompt_key] = (llm_response_time, len(llm_response))

        # if the last insert caused an eviction due to reaching maximum capacity
        if len(self._ap_cache) == self._max_size and self._ap_cache.last_evicted is not None:
            prompt_key, _ = self._ap_cache.last_evicted
            self._requests_db.remove(prompt_key)
            self._responses_db.remove_by_request(prompt_key)

        self._requests_db.save(
            EmbeddedRequestRecord(key=prompt_key, vector=self._embedder(prompt))
        )
        response_key = self._generate_key(llm_response)
        self._responses_db.save(
            ResponseRecord(key=response_key, request_key=prompt_key, response=llm_response)
        )
