"""SAGE wired through the same EchoLLM and Ollama APIs as cache_example.py."""

from sentence_transformers import SentenceTransformer

from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod
from echollm import EchoLLM
from llm import Ollama
from llm.ollama_llm import OllamaModel


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    vector = model.encode(text, normalize_embeddings=False)
    return vector.tolist()


def run_sage_example() -> None:
    echo_llm = EchoLLM(
        cache=SAGESimilarityCache(
            max_size=1_000,
            hit_distance_threshold=0.18,
            prompt_embedder=embed,
            ranking_distance_method=RankingDistanceMethod.COSINE,
            ghost_capacity=4_096,
            decay_half_life_requests=10_000,
            storage_path=".cache/sage.sqlite3",
            storage_namespace="example",
        ),
        llm=Ollama(
            model=OllamaModel.GEMMA3_1B,
            host="http://localhost:11434",
        ),
    )
    print(echo_llm.ask("How can I reset my password?"))


if __name__ == "__main__":
    run_sage_example()
