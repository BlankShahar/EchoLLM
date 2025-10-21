import logging

from cache import ICache
from llm import ILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EchoLLM')


class EchoLLM:
    def __init__(self, cache: ICache, llm: ILLM):
        self._cache = cache
        self._llm = llm

    def ask(self, prompt: str) -> str:
        if self._cache.is_hit(prompt):
            logger.info('Cache Hit', extra={'prompt': prompt})
            return self._cache.on_hit(prompt)
        else:
            logger.info('Cache Miss', extra={'prompt': prompt})
            llm_response = self._llm.ask(prompt)
            logger.info(f'LLM response took {llm_response.time:.2f}ms')
            self._cache.on_miss(prompt, llm_response.response, llm_response_time=llm_response.time)
            return llm_response.response
