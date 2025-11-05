from cache.lru_similarity_cache import LRUSimilarityCache
from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.storage_client.faiss_client import FaissDistanceMethod
from echo_llm import EchoLLM
from llm import Ollama
from llm.ollama_llm import OllamaModel
from text_similarity import text_embedder


def run_cache_example():
    echo_llm = EchoLLM(
        cache=LRUSimilarityCache(
            max_size=10,
            hit_distance_threshold=0.2,
            candidates_number=10,
            ranking_distance_method=RankingDistanceMethod.COSINE,
            db_distance_method=FaissDistanceMethod.L2,
            prompt_embedder=text_embedder.sbert_embedder,
        ),
        llm=Ollama(
            model=OllamaModel.GEMMA3_1B,
            host='http://localhost:11434',
        )
    )

    # First prompt must be a Miss
    request1 = 'Write me a short script of calculator in python'
    response1 = echo_llm.ask(request1)
    print(response1)
    print('-------------')

    # Suppose to be a Hit
    request2 = 'Make a simple calculator in python'
    response2 = echo_llm.ask(request2)
    print(response2)
    print('-------------')

    # Suppose to be a Miss
    request3 = 'Hi'
    response3 = echo_llm.ask(request3)
    print(response3)


if __name__ == '__main__':
    run_cache_example()
