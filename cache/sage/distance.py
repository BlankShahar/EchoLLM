import numpy as np

from cache.similarity_cache import RankingDistanceMethod


class VectorSpace:
    """Vector preparation and exact vectorized distance calculations."""

    def __init__(self, method: RankingDistanceMethod):
        self.method = method

    def prepare(self, vector: list[float] | np.ndarray) -> np.ndarray:
        prepared = np.asarray(vector, dtype=np.float32)
        if prepared.ndim != 1 or prepared.size == 0:
            raise ValueError("Embedding must be a non-empty one-dimensional vector")
        if not np.isfinite(prepared).all():
            raise ValueError("Embedding contains NaN or infinite values")
        if self.method == RankingDistanceMethod.COSINE:
            norm = float(np.linalg.norm(prepared))
            if norm > 0.0:
                prepared = prepared / norm
        return np.ascontiguousarray(prepared, dtype=np.float32)

    def distances(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: matrix={matrix.shape[1]}, vector={vector.shape[0]}"
            )

        if self.method == RankingDistanceMethod.COSINE:
            distances = 1.0 - matrix @ vector
            zero_rows = np.linalg.norm(matrix, axis=1) == 0.0
            vector_is_zero = float(np.linalg.norm(vector)) == 0.0
            if vector_is_zero:
                distances[:] = 1.0
                distances[zero_rows] = 0.0
            else:
                distances[zero_rows] = 1.0
            return np.clip(distances, 0.0, 2.0).astype(np.float32, copy=False)

        difference = matrix - vector
        if self.method == RankingDistanceMethod.EUCLIDEAN:
            return np.linalg.norm(difference, axis=1).astype(np.float32, copy=False)
        if self.method == RankingDistanceMethod.MANHATTAN:
            return np.abs(difference).sum(axis=1).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported distance method: {self.method}")

    def covers(self, matrix: np.ndarray, vector: np.ndarray, threshold: float) -> np.ndarray:
        return self.distances(matrix, vector) <= threshold

    def utilities(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        threshold: float,
        *,
        soft: bool,
        power: float,
    ) -> np.ndarray:
        """Return scoring utility without changing the binary cache-hit rule."""
        distances = self.distances(matrix, vector)
        if not soft:
            return (distances <= threshold).astype(np.float32)
        if threshold == 0.0:
            return np.isclose(distances, 0.0, rtol=0.0, atol=1e-7).astype(np.float32)
        utilities = np.clip(1.0 - distances / threshold, 0.0, 1.0)
        if power != 1.0:
            utilities = np.power(utilities, power)
        utilities[distances > threshold] = 0.0
        return utilities.astype(np.float32, copy=False)
