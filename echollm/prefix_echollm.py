import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Optional, Iterator

from jinja2 import Template

from cache.prefix_based.prefix_similarity_cache import IPrefixSimilarityCache
from llm import ILLM

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

    def stream_ask(self, prompt: str, force_llm: bool = False) -> Iterator[str]:
        if self._cache is None or force_llm:
            return self._stream_ask_llm(prompt)

        if self._cache.is_hit(prompt):
            logger.info('Cache Hit', extra={'prompt': prompt})
            # query the cache and ask the llm simultaneously
            prefix_response = self._cache.on_hit(prompt, retrieve_only=True)
            prefix_prompt = Template(
                (_CWD.parent / 'cache' / 'prefix_based' / 'prompt_template.j2').read_text()
            ).render(prompt=prompt, prefix=prefix_response)
            llm_stream = self._stream_ask_llm(prefix_prompt, False, True)
            return chain([prefix_prompt], llm_stream)
        else:
            logger.info('Cache Miss', extra={'prompt': prompt})
            return self._stream_ask_llm(prompt, True, False)

    def _stream_ask_llm(
            self, prompt: str, is_on_miss_event: bool = False, should_update_item_stats: bool = True
    ) -> Iterator[str]:
        if is_on_miss_event and should_update_item_stats:
            raise ValueError(
                '`is_on_miss` and `should_update_item_stats` are mutually exclusive! (on_miss already update item stats)'
            )

        llm_stream = self._llm.stream_ask(prompt)
        chunk, full_response, llm_delay = None, '', None
        for chunk in llm_stream:
            if chunk.is_first:
                llm_delay = chunk.delay
                logger.info(f'LLM first token response took {llm_delay:.2f}ms')
                if self._cache and should_update_item_stats:
                    self._cache.update_item_stats(prompt, llm_delay=chunk.delay)
            full_response += chunk.response_chunk
            yield chunk.response_chunk
        logger.info(f'LLM full response took {chunk.delay:.2f}ms')

        if is_on_miss_event:
            self._cache.on_miss(prompt, full_response, llm_delay=llm_delay)
