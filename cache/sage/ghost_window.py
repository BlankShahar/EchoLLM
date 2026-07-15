import numpy as np


class GhostWindow:
    """Bounded recent-demand window with resident coverage bookkeeping.

    The buffer stores request embeddings and whether each resident slot covers each
    request. It never stores responses and can never serve a cache hit.
    """

    def __init__(self, capacity: int, resident_capacity: int):
        if capacity <= 0 or resident_capacity <= 0:
            raise ValueError("capacities must be positive")
        self._capacity = capacity
        self._resident_capacity = resident_capacity
        self._vectors: np.ndarray | None = None
        self._steps = np.zeros(capacity, dtype=np.int64)
        self._ids = np.full(capacity, -1, dtype=np.int64)
        self._active = np.zeros(capacity, dtype=np.bool_)
        self._coverage = np.zeros((capacity, resident_capacity), dtype=np.bool_)
        self._coverage_counts = np.zeros(capacity, dtype=np.int32)
        self._coverage_xor = np.zeros(capacity, dtype=np.int64)
        self._next_slot = 0
        self._next_id = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, vector: np.ndarray, step: int, coverage: np.ndarray) -> int:
        if coverage.shape != (self._resident_capacity,):
            raise ValueError("coverage vector has an invalid shape")
        self._ensure_dimension(vector.shape[0])
        assert self._vectors is not None

        physical_slot = self._next_slot
        observation_id = self._next_id
        self._next_id += 1

        self._vectors[physical_slot] = vector
        self._steps[physical_slot] = step
        self._ids[physical_slot] = observation_id
        self._coverage[physical_slot] = coverage
        covering_slots = np.flatnonzero(coverage).astype(np.int64, copy=False)
        self._coverage_counts[physical_slot] = int(covering_slots.size)
        self._coverage_xor[physical_slot] = np.bitwise_xor.reduce(
            covering_slots,
            initial=np.int64(0),
        )
        if not self._active[physical_slot]:
            self._active[physical_slot] = True
            self._size += 1

        self._next_slot = (physical_slot + 1) % self._capacity
        return observation_id

    def active_vectors(self) -> np.ndarray:
        if self._vectors is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._vectors[self._active]

    def active_steps(self) -> np.ndarray:
        return self._steps[self._active]

    def active_coverage(self) -> np.ndarray:
        return self._coverage[self._active]

    def active_coverage_counts(self) -> np.ndarray:
        return self._coverage_counts[self._active]

    def active_unique_owners(self) -> np.ndarray:
        """Return the sole covering slot for rows whose coverage count is one."""
        return self._coverage_xor[self._active]

    def weights(
        self,
        current_step: int,
        half_life_requests: float | None,
        *,
        current_observation_id: int | None = None,
        current_observation_weight: float = 1.0,
    ) -> np.ndarray:
        steps = self.active_steps()
        if half_life_requests is None:
            weights = np.ones(steps.shape[0], dtype=np.float64)
        else:
            ages = np.maximum(0, current_step - steps).astype(np.float64)
            weights = np.exp2(-ages / half_life_requests)
        if current_observation_id is not None and current_observation_weight != 1.0:
            active_ids = self._ids[self._active]
            weights[active_ids == current_observation_id] *= current_observation_weight
        return weights

    def replace_resident_column(self, resident_slot: int, new_coverage: np.ndarray) -> None:
        active_count = self._size
        if new_coverage.shape != (active_count,):
            raise ValueError(
                f"new_coverage must contain {active_count} values, got {new_coverage.shape}"
            )
        active_indices = np.flatnonzero(self._active)
        old = self._coverage[active_indices, resident_slot]
        changed = old ^ new_coverage
        self._coverage_counts[active_indices] += new_coverage.astype(np.int32) - old.astype(np.int32)
        self._coverage_xor[active_indices[changed]] ^= np.int64(resident_slot)
        self._coverage[active_indices, resident_slot] = new_coverage

    def clear_resident_column(self, resident_slot: int) -> None:
        active_indices = np.flatnonzero(self._active)
        old = self._coverage[active_indices, resident_slot]
        self._coverage_counts[active_indices] -= old.astype(np.int32)
        self._coverage_xor[active_indices[old]] ^= np.int64(resident_slot)
        self._coverage[active_indices, resident_slot] = False

    def coverage_column(self, resident_slot: int) -> np.ndarray:
        return self._coverage[self._active, resident_slot].copy()

    def _ensure_dimension(self, dimension: int) -> None:
        if self._vectors is None:
            self._vectors = np.empty((self._capacity, dimension), dtype=np.float32)
            return
        if self._vectors.shape[1] != dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._vectors.shape[1]} to {dimension}"
            )
