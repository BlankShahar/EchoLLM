from cache import ICache
from llm import ILLM


class EchoLLM:
    def __init__(self, cache: ICache, llm: ILLM):
        self._cache = cache
        self._llm = llm

    def ask(self, prompt: str) -> str:
        cache_response = self._cache.on_request(prompt)
        return cache_response if cache_response else self._llm.ask(prompt).response
