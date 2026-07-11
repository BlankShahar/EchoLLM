import math
from typing import Iterable

import numpy as np


def euclidean_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    left = np.asarray(tuple(vector1), dtype=np.float32)
    right = np.asarray(tuple(vector2), dtype=np.float32)
    return float(np.linalg.norm(left - right))


def manhattan_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    left = np.asarray(tuple(vector1), dtype=np.float32)
    right = np.asarray(tuple(vector2), dtype=np.float32)
    return float(np.abs(left - right).sum())


def cosine_distance(vector1: Iterable[float], vector2: Iterable[float]) -> float:
    left = np.asarray(tuple(vector1), dtype=np.float32)
    right = np.asarray(tuple(vector2), dtype=np.float32)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(1.0 - max(-1.0, min(1.0, similarity)))


def normalize(vector: Iterable[float]) -> list[float]:
    array = np.asarray(tuple(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if math.isclose(norm, 0.0):
        return array.tolist()
    return (array / norm).tolist()
