import logging
from typing import Optional

from cache import ICache
from llm import ILLM, LLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EchoLLM')


class EchoLLM:
    def __init__(self, cache: Optional[ICache], llm: ILLM):
        self._cache = cache
        self._llm = llm

        if cache is None:
            logger.info('No Cache -- Asking LLM')
        else:
            logger.info(f'Initiated Cache - `{self._cache.policy_name}`')

    def ask(self, prompt: str, force_llm: bool = False) -> str:
        if self._cache is None or force_llm:
            return self._ask_llm(prompt).response

        if self._cache.is_hit(prompt):
            logger.info('Cache Hit', extra={'prompt': prompt})
            return self._cache.on_hit(prompt)
        else:
            logger.info('Cache Miss', extra={'prompt': prompt})
            llm_response = self._ask_llm(prompt)
            self._cache.on_miss(prompt, llm_response.response, llm_latency=llm_response.latency)
            return llm_response.response

    def _ask_llm(self, prompt: str) -> LLMResponse:
        llm_response = self._llm.ask(prompt)
        logger.info(f'LLM response took {llm_response.latency:.2f}ms')
        return llm_response
