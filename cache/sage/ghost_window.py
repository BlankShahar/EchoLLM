import numpy as np


class GhostWindow:
    """Bounded demand evidence with soft resident-utility bookkeeping."""

    def __init__(self, capacity: int, resident_capacity: int):
        if capacity <= 0 or resident_capacity <= 0:
            raise ValueError("capacities must be positive")
        self._capacity = capacity
        self._resident_capacity = resident_capacity
        self._vectors: np.ndarray | None = None
        self._steps = np.zeros(capacity, dtype=np.int64)
        self._ids = np.full(capacity, -1, dtype=np.int64)
        self._active = np.zeros(capacity, dtype=np.bool_)
        self._utilities = np.zeros((capacity, resident_capacity), dtype=np.float32)
        self._utility_sums = np.zeros(capacity, dtype=np.float64)
        self._nonzero_counts = np.zeros(resident_capacity, dtype=np.int32)
        self._top_slots = np.full((capacity, 2), -1, dtype=np.int64)
        self._top_values = np.zeros((capacity, 2), dtype=np.float32)
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
        observation_id, _ = self.add_with_evicted(vector, step, coverage)
        return observation_id

    def add_with_evicted(
        self,
        vector: np.ndarray,
        step: int,
        utilities: np.ndarray,
    ) -> tuple[int, tuple[np.ndarray, int, np.ndarray] | None]:
        if utilities.shape != (self._resident_capacity,):
            raise ValueError("utility vector has an invalid shape")
        self._ensure_dimension(vector.shape[0])
        assert self._vectors is not None

        physical_slot = self._next_slot
        evicted = None
        if self._active[physical_slot]:
            evicted = (
                self._vectors[physical_slot].copy(),
                int(self._steps[physical_slot]),
                self._utilities[physical_slot].copy(),
            )
            self._nonzero_counts -= (self._utilities[physical_slot] > 0.0).astype(
                np.int32
            )

        observation_id = self._next_id
        self._next_id += 1
        self._vectors[physical_slot] = vector
        self._steps[physical_slot] = step
        self._ids[physical_slot] = observation_id
        self._utilities[physical_slot] = np.asarray(utilities, dtype=np.float32)
        self._utility_sums[physical_slot] = float(
            self._utilities[physical_slot].sum(dtype=np.float64)
        )
        self._nonzero_counts += (self._utilities[physical_slot] > 0.0).astype(
            np.int32
        )
        self._refresh_top(np.asarray([physical_slot], dtype=np.int64))
        if not self._active[physical_slot]:
            self._active[physical_slot] = True
            self._size += 1

        self._next_slot = (physical_slot + 1) % self._capacity
        return observation_id, evicted

    def active_vectors(self) -> np.ndarray:
        if self._vectors is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._vectors[self._active]

    def active_steps(self) -> np.ndarray:
        return self._steps[self._active]

    def active_utilities(self) -> np.ndarray:
        return self._utilities[self._active]

    def active_top_values(self) -> np.ndarray:
        return self._top_values[self._active]

    def active_top_slots(self) -> np.ndarray:
        return self._top_slots[self._active]

    # Compatibility views retained for the original binary SAGE tests/API.
    def active_coverage(self) -> np.ndarray:
        return self.active_utilities() > 0.0

    def active_coverage_counts(self) -> np.ndarray:
        return np.count_nonzero(self.active_utilities() > 0.0, axis=1).astype(np.int32)

    def active_unique_owners(self) -> np.ndarray:
        coverage = self.active_coverage()
        return np.bitwise_xor.reduce(
            np.where(coverage, np.arange(self._resident_capacity), 0),
            axis=1,
            initial=np.int64(0),
        )

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

    def replace_resident_column(self, resident_slot: int, new_utilities: np.ndarray) -> None:
        if new_utilities.shape != (self._size,):
            raise ValueError(
                f"new_utilities must contain {self._size} values, got {new_utilities.shape}"
            )
        active_indices = np.flatnonzero(self._active)
        old_values = self._utilities[active_indices, resident_slot].astype(
            np.float64, copy=True
        )
        old_top = self._top_slots[active_indices]
        affected = active_indices[
            (old_top[:, 0] == resident_slot) | (old_top[:, 1] == resident_slot)
        ]
        unaffected = active_indices[
            (old_top[:, 0] != resident_slot) & (old_top[:, 1] != resident_slot)
        ]
        self._utilities[active_indices, resident_slot] = np.asarray(
            new_utilities, dtype=np.float32
        )
        self._nonzero_counts[resident_slot] += int(
            np.count_nonzero(self._utilities[active_indices, resident_slot] > 0.0)
            - np.count_nonzero(old_values > 0.0)
        )
        self._utility_sums[active_indices] += (
            self._utilities[active_indices, resident_slot].astype(np.float64, copy=False)
            - old_values
        )
        if affected.size:
            self._refresh_top(affected)
        if unaffected.size:
            values = self._utilities[unaffected, resident_slot]
            top1 = self._top_values[unaffected, 0]
            top2 = self._top_values[unaffected, 1]
            becomes_first = values > top1
            if bool(becomes_first.any()):
                rows = unaffected[becomes_first]
                self._top_values[rows, 1] = self._top_values[rows, 0]
                self._top_slots[rows, 1] = self._top_slots[rows, 0]
                self._top_values[rows, 0] = values[becomes_first]
                self._top_slots[rows, 0] = resident_slot
            becomes_second = (~becomes_first) & (values > top2)
            if bool(becomes_second.any()):
                rows = unaffected[becomes_second]
                self._top_values[rows, 1] = values[becomes_second]
                self._top_slots[rows, 1] = resident_slot

    def clear_resident_column(self, resident_slot: int) -> None:
        self.replace_resident_column(
            resident_slot, np.zeros(self._size, dtype=np.float32)
        )

    def coverage_column(self, resident_slot: int) -> np.ndarray:
        return (self._utilities[self._active, resident_slot] > 0.0).copy()

    def utility_column(self, resident_slot: int) -> np.ndarray:
        return self._utilities[self._active, resident_slot].copy()

    def nonzero_counts(self, resident_slots: np.ndarray) -> np.ndarray:
        return self._nonzero_counts[resident_slots].copy()

    def fractional_responsibility(
        self,
        weights: np.ndarray,
        resident_slots: np.ndarray,
    ) -> np.ndarray:
        if weights.shape != (self._size,):
            raise ValueError("weights shape mismatch")
        if resident_slots.size == 0:
            return np.empty(0, dtype=np.float64)
        active_indices = np.flatnonzero(self._active)
        row_totals = self._utility_sums[active_indices]
        shares = np.divide(
            weights,
            row_totals,
            out=np.zeros_like(weights, dtype=np.float64),
            where=row_totals > 0.0,
        )
        responsibility = np.empty(resident_slots.size, dtype=np.float64)
        chunk_size = 256
        for start in range(0, resident_slots.size, chunk_size):
            stop = min(start + chunk_size, resident_slots.size)
            chunk = resident_slots[start:stop]
            selected = self._utilities[np.ix_(active_indices, chunk)]
            responsibility[start:stop] = shares @ selected
        return responsibility

    def _refresh_top(self, physical_indices: np.ndarray) -> None:
        if physical_indices.size == 0:
            return
        matrix = self._utilities[physical_indices]
        if self._resident_capacity == 1:
            self._top_slots[physical_indices, 0] = 0
            self._top_slots[physical_indices, 1] = -1
            self._top_values[physical_indices, 0] = matrix[:, 0]
            self._top_values[physical_indices, 1] = 0.0
            return
        candidates = np.argpartition(matrix, -2, axis=1)[:, -2:]
        values = np.take_along_axis(matrix, candidates, axis=1)
        order = np.argsort(values, axis=1)[:, ::-1]
        sorted_slots = np.take_along_axis(candidates, order, axis=1)
        sorted_values = np.take_along_axis(values, order, axis=1)
        self._top_slots[physical_indices] = sorted_slots
        self._top_values[physical_indices] = sorted_values

    def _ensure_dimension(self, dimension: int) -> None:
        if self._vectors is None:
            self._vectors = np.empty((self._capacity, dimension), dtype=np.float32)
            return
        if self._vectors.shape[1] != dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._vectors.shape[1]} to {dimension}"
            )
