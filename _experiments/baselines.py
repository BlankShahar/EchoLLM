import threading
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from cache import CacheLookup, ICache
from cache.sage.distance import VectorSpace
from cache.similarity_cache import RankingDistanceMethod


class BaselineKind(StrEnum):
    LRU = "LRU"
    LFU = "LFU"
    FIFO = "FIFO"
    RR = "RR"


class BaselineLookupContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: list[float]


class BaselineStats(BaseModel):
    requests: int = 0
    hits: int = 0
    misses: int = 0
    admissions: int = 0
    evictions: int = 0


class ExactSemanticBaselineCache(ICache):
    """Exact semantic-cache baselines sharing one lookup implementation.

    The experiment deliberately uses exact scans for every policy so the only
    changing variable is admission/eviction, not approximate-search recall.
    """

    def __init__(
        self,
        kind: BaselineKind,
        max_size: int,
        hit_distance_threshold: float,
        prompt_embedder: Callable[[str], Sequence[float]],
        *,
        distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
        seed: int = 7,
    ) -> None:
        super().__init__(max_size, kind.value)
        self.kind = kind
        self._threshold = hit_distance_threshold
        self._embedder = prompt_embedder
        self._space = VectorSpace(distance_method)
        self._rng = np.random.default_rng(seed)
        self._vectors: np.ndarray | None = None
        self._active = np.zeros(max_size, dtype=np.bool_)
        self._prompts: list[str | None] = [None] * max_size
        self._responses: list[str | None] = [None] * max_size
        self._inserted = np.zeros(max_size, dtype=np.int64)
        self._last_access = np.zeros(max_size, dtype=np.int64)
        self._frequency = np.zeros(max_size, dtype=np.int64)
        self._step = 0
        self._stats = BaselineStats()
        self._lock = threading.RLock()
        self._legacy_local = threading.local()

    def lookup(self, request: str) -> CacheLookup:
        vector = self._space.prepare(list(self._embedder(request)))
        with self._lock:
            self._ensure_dimension(vector.shape[0])
            self._step += 1
            distances = np.full(self._max_size, np.inf, dtype=np.float32)
            active_slots = np.flatnonzero(self._active)
            if active_slots.size:
                assert self._vectors is not None
                distances[active_slots] = self._space.distances(self._vectors[active_slots], vector)
            coverage = self._active & (distances <= self._threshold)
            self._stats.requests += 1
            context = BaselineLookupContext(embedding=vector.tolist())
            if not coverage.any():
                self._stats.misses += 1
                return CacheLookup(hit=False, context=context)

            slot = int(np.argmin(np.where(coverage, distances, np.inf)))
            self._last_access[slot] = self._step
            self._frequency[slot] += 1
            self._stats.hits += 1
            response = self._responses[slot]
            if response is None:
                raise RuntimeError(f"Active slot {slot} has no response")
            return CacheLookup(
                hit=True,
                response=response,
                context=context,
                metadata={
                    "prompt_distance": float(distances[slot]),
                    "matched_prompt": self._prompts[slot],
                    "resident_slot": slot,
                },
            )

    def is_hit(self, request: str) -> bool:
        lookup = self.lookup(request)
        self._legacy_local.lookup = (request, lookup)
        return lookup.hit

    def on_hit(self, request: str, **kwargs: Any) -> str:
        cached = getattr(self._legacy_local, "lookup", None)
        if cached is not None and cached[0] == request and cached[1].hit:
            self._legacy_local.lookup = None
            return str(cached[1].response)
        lookup = self.lookup(request)
        if not lookup.hit:
            raise KeyError("Request is not covered")
        return str(lookup.response)

    def on_miss(self, request: str, response: str, **kwargs: Any) -> None:
        raw_context = kwargs.get("lookup_context")
        if raw_context is None:
            vector = self._space.prepare(list(self._embedder(request)))
        else:
            vector = np.asarray(BaselineLookupContext.model_validate(raw_context).embedding, dtype=np.float32)
        with self._lock:
            self._ensure_dimension(vector.shape[0])
            # Recheck to avoid duplicates if miss completions race.
            active_slots = np.flatnonzero(self._active)
            if active_slots.size:
                assert self._vectors is not None
                distances = self._space.distances(self._vectors[active_slots], vector)
                if bool((distances <= self._threshold).any()):
                    return

            free = np.flatnonzero(~self._active)
            if free.size:
                slot = int(free[0])
            else:
                slot = self._victim_slot()
                self._stats.evictions += 1
            assert self._vectors is not None
            self._vectors[slot] = vector
            self._active[slot] = True
            self._prompts[slot] = request
            self._responses[slot] = response
            self._inserted[slot] = self._step
            self._last_access[slot] = self._step
            self._frequency[slot] = 1
            self._stats.admissions += 1

    @property
    def current_size(self) -> int:
        return int(np.count_nonzero(self._active))

    def stats(self) -> BaselineStats:
        return self._stats.model_copy(deep=True)

    def _victim_slot(self) -> int:
        active = np.flatnonzero(self._active)
        if self.kind == BaselineKind.LRU:
            return int(active[np.argmin(self._last_access[active])])
        if self.kind == BaselineKind.FIFO:
            return int(active[np.argmin(self._inserted[active])])
        if self.kind == BaselineKind.LFU:
            minimum = self._frequency[active].min()
            tied = active[self._frequency[active] == minimum]
            return int(tied[np.argmin(self._last_access[tied])])
        if self.kind == BaselineKind.RR:
            return int(self._rng.choice(active))
        raise ValueError(f"Unsupported baseline kind: {self.kind}")

    def _ensure_dimension(self, dimension: int) -> None:
        if self._vectors is None:
            self._vectors = np.zeros((self._max_size, dimension), dtype=np.float32)
        elif self._vectors.shape[1] != dimension:
            raise ValueError("Embedding dimension changed")
