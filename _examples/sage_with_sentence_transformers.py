"""Minimal SAGE construction example.

Install sentence-transformers before running. Supply a concrete EchoLLM ILLM
backend for `llm`.
"""

from sentence_transformers import SentenceTransformer

from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod
from echollm import EchoLLM


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    vector = model.encode(text, normalize_embeddings=False)
    return vector.tolist()


cache = SAGESimilarityCache(
    max_size=1_000,
    hit_distance_threshold=0.18,
    prompt_embedder=embed,
    ranking_distance_method=RankingDistanceMethod.COSINE,
    ghost_capacity=4_096,
    decay_half_life_requests=10_000,
    storage_path=".cache/sage.sqlite3",
    storage_namespace="example",
)

# echo = EchoLLM(cache=cache, llm=llm)
# print(echo.ask("How can I reset my password?"))
