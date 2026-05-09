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
def cosine_similarity(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    """
    Exact cosine similarity between two vectors.

    Returns 0.0 when either vector is the zero vector.
    """
    arr_a = np.asarray(vector1, dtype=np.float64)
    arr_b = np.asarray(vector2, dtype=np.float64)
    normalised_a = float(np.linalg.norm(arr_a))
    normalised_b = float(np.linalg.norm(arr_b))
    if normalised_a == 0.0 or normalised_b == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (normalised_a * normalised_b))


@lru_cache
def cosine_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    return 1 - cosine_similarity(vector1, vector2)


@lru_cache
def normalize(vector: Iterable[float]) -> Iterable[float]:
    """L2-normalize a vector. If it's all zeros, return as-is."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()
