import math
from functools import lru_cache
from typing import Iterable

import numpy as np


@lru_cache
def euclidean_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))


@lru_cache
def manhattan_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    return sum(abs(a - b) for a, b in zip(vector1, vector2))


@lru_cache
def cosine_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(b * b for b in vector2))

    if norm1 == 0 and norm2 == 0:
        return 0.0
    if norm1 == 0 or norm2 == 0:
        return 1.0  # convention: undefined → maximally distant

    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity


@lru_cache
def normalize(vector: Iterable[float]) -> Iterable[float]:
    """L2-normalize a vector. If it's all zeros, return as-is."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()
