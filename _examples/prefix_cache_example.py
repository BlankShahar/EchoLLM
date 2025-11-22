from cache.prefix_based.prefix_lru_similarity_cache import PrefixLRUSimilarityCache
from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.storage_client.faiss_client import FaissDistanceMethod
from llm import Ollama
from llm.ollama_llm import OllamaModel
from echollm.prefix_echollm import PrefixEchoLLM
from text_similarity import text_embedder


def run_cache_example():
    echo_llm = PrefixEchoLLM(
        cache=PrefixLRUSimilarityCache(
            max_size=10,
            hit_distance_threshold=0.2,
            candidates_number=10,
            ranking_distance_method=RankingDistanceMethod.COSINE,
            db_distance_method=FaissDistanceMethod.L2,
            prompt_embedder=text_embedder.sbert_embedder,
            bandwidth=7_500,  # Mbps
            delay_ewma_smoothing_factor=0.2,
            prefix_size_confidence_factor=2
        ),
        llm=Ollama(
            model=OllamaModel.GEMMA3_1B,
            host='http://localhost:11434',
        )
    )

    # First prompt must be a Miss
    request1 = 'Show me 2 ways to sort a list of numbers in python'
    response1 = echo_llm.stream_ask(request1)
    print(response1)
    print('-------------')

    # Suppose to be a Hit
    request2 = 'Explain in 2 approaches how to sort a list in python'
    response2 = echo_llm.stream_ask(request2)
    print(response2)
    print('-------------')

    # Suppose to be a Miss
    request3 = 'Hi'
    response3 = echo_llm.stream_ask(request3)
    print(response3)


if __name__ == '__main__':
    run_cache_example()
