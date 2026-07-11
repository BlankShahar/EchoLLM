import logging
from typing import Optional

from cache import ICache
from llm import ILLM, LLMResponse

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
        if self._cache is None or force_llm:
            return self._ask_llm(prompt).response

        lookup = self._cache.lookup(prompt)
        if lookup.hit:
            logger.info("Cache Hit", extra={"prompt": prompt})
            return str(lookup.response)

        logger.info("Cache Miss", extra={"prompt": prompt})
        llm_response = self._ask_llm(prompt)
        self._cache.on_miss(
            prompt,
            llm_response.response,
            llm_latency=llm_response.latency,
            lookup_context=lookup.context,
        )
        return llm_response.response

    def _ask_llm(self, prompt: str) -> LLMResponse:
        llm_response = self._llm.ask(prompt)
        logger.info("LLM response took %.2fms", llm_response.latency)
        return llm_response
