import logging

from cache import ICache
from llm import ILLM

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('EchoLLM')


class EchoLLM:
    def __init__(self, cache: ICache, llm: ILLM):
        self._cache = cache
        self._llm = llm

    def ask(self, prompt: str) -> str:
        if self._cache.is_hit(prompt):
            logger.debug('Cache Hit', extra={'prompt': prompt})
            return self._cache.on_hit(prompt)
        else:
            logger.debug('Cache Miss', extra={'prompt': prompt})
            llm_response = self._llm.ask(prompt)
            logger.debug(f'LLM response took {llm_response:.2f}ms')
            return llm_response.response
