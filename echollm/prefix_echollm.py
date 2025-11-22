import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from jinja2 import Template

from cache.prefix_based.prefix_similarity_cache import IPrefixSimilarityCache
from llm import ILLM
from llm.illm import StreamedLLMResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EchoLLM')

_CWD = Path(__file__).parent


class PrefixEchoLLM:
    def __init__(self, cache: Optional[IPrefixSimilarityCache], llm: ILLM):
        self._cache = cache
        self._llm = llm

        if cache is None:
            logger.info('No Cache -- Asking LLM')
        else:
            logger.info(f'Initiated Cache - `{self._cache.policy_name}`')

    def stream_ask(self, prompt: str, force_llm: bool = False) -> str:
        if self._cache is None:
            return self._llm.stream_ask(prompt).response
        if force_llm:  # and cache exist
            llm_response = self._stream_ask_llm(prompt)
            self._cache.update_item_stats(prompt, llm_delay=llm_response.delay)
            return llm_response.response

        if self._cache.is_hit(prompt):
            logger.info('Cache Hit', extra={'prompt': prompt})
            with ThreadPoolExecutor(max_workers=1) as executor:
                # query the cache and ask the llm simultaneously
                prefix_response = self._cache.on_hit(prompt, retrieve_only=True)
                prefix_prompt = Template(
                    (_CWD.parent / 'cache' / 'prefix_based' / 'prompt_template.j2').read_text()
                ).render(prompt=prompt, prefix=prefix_response)
                llm_response: StreamedLLMResponse = executor.submit(self._stream_ask_llm, prefix_prompt).result()
                self._cache.on_hit(prompt, llm_delay=llm_response.delay)

                return f"{prefix_response}{llm_response.response}"
        else:
            logger.info('Cache Miss', extra={'prompt': prompt})
            llm_response = self._stream_ask_llm(prompt)
            self._cache.on_miss(prompt, llm_response.response, llm_delay=llm_response.delay)
            return llm_response.response

    def _stream_ask_llm(self, prompt: str) -> StreamedLLMResponse:
        llm_response = self._llm.stream_ask(prompt)
        logger.info(f'LLM first token response took {llm_response.delay:.2f}ms')
        logger.info(f'LLM full response took {llm_response.latency:.2f}ms')
        return llm_response
