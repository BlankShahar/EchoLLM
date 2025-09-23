import math


def euclidean_distance(vector1: list[float], vector2: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))


def manhattan_distance(vector1: list[float], vector2: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(vector1, vector2))


def cosine_distance(vector1: list[float], vector2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(b * b for b in vector2))

    if norm1 == 0 and norm2 == 0:
        return 0.0
    if norm1 == 0 or norm2 == 0:
        return 1.0  # convention: undefined → maximally distant

    cosine_similarity = dot_product / (norm1 * norm2)
    return 1 - cosine_similarity
