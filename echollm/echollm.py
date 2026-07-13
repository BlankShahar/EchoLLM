import logging
from time import perf_counter_ns
from typing import Optional

from cache import ICache
from llm import ILLM, LLMResponse

from .models import EchoLLMResponse

logger = logging.getLogger("EchoLLM")


class EchoLLM:
    def __init__(self, cache: Optional[ICache], llm: ILLM):
        self._cache = cache
        self._llm = llm
        if cache is None:
            logger.info("No cache configured")
        else:
            logger.info(f'Initiated Cache - `{self._cache.policy_name}`')

    def ask(self, prompt: str, force_llm: bool = False) -> str:
        return self.ask_with_metadata(prompt, force_llm=force_llm).response

    def ask_with_metadata(self, prompt: str, force_llm: bool = False) -> EchoLLMResponse:
        if self._cache is None or force_llm:
            llm_response = self._ask_llm(prompt)
            return EchoLLMResponse(
                response=llm_response.response,
                cache_hit=False,
                llm_latency=llm_response.latency,
                cache_lookup_latency=0.0,
                cache_admission_latency=0.0,
            )

        lookup_started = perf_counter_ns()
        lookup = self._cache.lookup(prompt)
        lookup_latency = _elapsed_ms(lookup_started)
        if lookup.hit:
            logger.info("Cache Hit", extra={"prompt": prompt})
            return EchoLLMResponse(
                response=str(lookup.response),
                cache_hit=True,
                llm_latency=0.0,
                cache_lookup_latency=lookup_latency,
                cache_admission_latency=0.0,
                cache_metadata=lookup.metadata,
            )

        logger.info("Cache Miss", extra={"prompt": prompt})
        llm_response = self._ask_llm(prompt)
        admission_started = perf_counter_ns()
        self._cache.on_miss(
            prompt,
            llm_response.response,
            llm_latency=llm_response.latency,
            lookup_context=lookup.context,
        )
        return EchoLLMResponse(
            response=llm_response.response,
            cache_hit=False,
            llm_latency=llm_response.latency,
            cache_lookup_latency=lookup_latency,
            cache_admission_latency=_elapsed_ms(admission_started),
            cache_metadata=lookup.metadata,
        )

    def _ask_llm(self, prompt: str) -> LLMResponse:
        llm_response = self._llm.ask(prompt)
        logger.info("LLM response took %.2fms", llm_response.latency)
        return llm_response


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000.0
