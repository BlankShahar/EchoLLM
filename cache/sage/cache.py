import hashlib
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from cache import CacheLookup, ICache
from cache.similarity_cache import RankingDistanceMethod

from .config import SAGEConfig
from .distance import VectorSpace
from .ghost_window import GhostWindow
from .models import (
    PersistedResident,
    SAGEDecision,
    SAGELookupContext,
    SAGEStats,
    SAGEStorageMetadata,
)
from .scorer import SAGEScorer
from .storage import NullSAGEStorage, SAGEStorage, SQLiteSAGEStorage


class SAGESimilarityCache(ICache):
    """Semantic Admission and Gain-based Eviction.

    SAGE maximizes recency-weighted semantic coverage over a bounded history of
    requests. A candidate is admitted only when replacing some resident strictly
    improves that objective. The implementation uses exact vectorized distances,
    a fixed resident-slot layout, and O(W + C) replacement scoring after the
    candidate-to-window distances are known.
    """

    def __init__(
        self,
        max_size: int,
        hit_distance_threshold: float,
        prompt_embedder: Callable[[str], Sequence[float]],
        *,
        ranking_distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
        ghost_capacity: int = 4096,
        decay_half_life_requests: float | None = None,
        admission_margin: float = 0.0,
        storage_path: str | Path | None = None,
        storage_namespace: str = "default",
        storage: SAGEStorage | None = None,
    ) -> None:
        config = SAGEConfig(
            max_size=max_size,
            hit_distance_threshold=hit_distance_threshold,
            distance_method=ranking_distance_method,
            ghost_capacity=ghost_capacity,
            decay_half_life_requests=decay_half_life_requests,
            admission_margin=admission_margin,
            storage_path=Path(storage_path) if storage_path is not None else None,
            storage_namespace=storage_namespace,
        )
        super().__init__(config.max_size, "SAGE")
        self.config = config
        self._embedder = prompt_embedder
        self._space = VectorSpace(config.distance_method)
        self._ghost = GhostWindow(config.ghost_capacity, config.max_size)
        self._scorer = SAGEScorer()
        self._lock = threading.RLock()
        self._legacy_local = threading.local()

        self._resident_vectors: np.ndarray | None = None
        self._active = np.zeros(config.max_size, dtype=np.bool_)
        self._keys: list[str | None] = [None] * config.max_size
        self._prompts: list[str | None] = [None] * config.max_size
        self._responses: list[str | None] = [None] * config.max_size
        self._inserted_steps = np.zeros(config.max_size, dtype=np.int64)
        self._last_access_steps = np.zeros(config.max_size, dtype=np.int64)
        self._step = 0
        self._stats = SAGEStats()
        self._last_decision: SAGEDecision | None = None
        self._storage_initialized = False

        if storage is not None and config.storage_path is not None:
            raise ValueError("Pass either storage or storage_path, not both")
        if storage is not None:
            self._storage = storage
        elif config.storage_path is not None:
            self._storage = SQLiteSAGEStorage(config.storage_path, config.storage_namespace)
        else:
            self._storage = NullSAGEStorage()

        self._restore_residents()

    def lookup(self, request: str) -> CacheLookup:
        started = perf_counter_ns()
        vector = self._space.prepare(list(self._embedder(request)))
        with self._lock:
            self._ensure_dimension(vector.shape[0])
            self._step += 1
            distances, coverage = self._resident_distances_and_coverage(vector)
            observation_id = self._ghost.add(vector, self._step, coverage)
            context = SAGELookupContext(
                embedding=vector.tolist(),
                observation_id=observation_id,
                request_step=self._step,
            )

            self._stats.requests += 1
            if bool(coverage.any()):
                masked_distances = np.where(coverage, distances, np.inf)
                slot = int(np.argmin(masked_distances))
                response = self._responses[slot]
                if response is None:
                    raise RuntimeError(f"Active resident slot {slot} has no response")
                self._last_access_steps[slot] = self._step
                self._stats.hits += 1
                result = CacheLookup(
                    hit=True,
                    response=response,
                    context=context,
                    metadata={
                        "cache_key": self._keys[slot],
                        "matched_prompt": self._prompts[slot],
                        "prompt_distance": float(distances[slot]),
                        "resident_slot": slot,
                    },
                )
            else:
                self._stats.misses += 1
                result = CacheLookup(hit=False, context=context)

        self._stats.lookup_time_ms += (perf_counter_ns() - started) / 1_000_000.0
        return result

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
            raise KeyError("Request is not covered by the cache")
        return str(lookup.response)

    def on_miss(self, request: str, response: str, **kwargs: Any) -> None:
        started = perf_counter_ns()
        raw_context = kwargs.get("lookup_context")
        if raw_context is None:
            cached = getattr(self._legacy_local, "lookup", None)
            if cached is not None and cached[0] == request and not cached[1].hit:
                raw_context = cached[1].context
                self._legacy_local.lookup = None

        if raw_context is None:
            vector = self._space.prepare(list(self._embedder(request)))
            with self._lock:
                self._ensure_dimension(vector.shape[0])
                self._step += 1
                _, coverage = self._resident_distances_and_coverage(vector)
                observation_id = self._ghost.add(vector, self._step, coverage)
                context = SAGELookupContext(
                    embedding=vector.tolist(),
                    observation_id=observation_id,
                    request_step=self._step,
                )
                self._stats.requests += 1
                self._stats.misses += 1
        else:
            context = SAGELookupContext.model_validate(raw_context)
            vector = np.asarray(context.embedding, dtype=np.float32)

        candidate_key = self._generate_key(request)
        with self._lock:
            self._ensure_dimension(vector.shape[0])
            _, current_coverage = self._resident_distances_and_coverage(vector)
            if bool(current_coverage.any()):
                self._stats.rejections += 1
                self._stats.concurrent_deduplications += 1
                self._last_decision = SAGEDecision(
                    admitted=False,
                    reason="concurrent_semantic_duplicate",
                    candidate_key=candidate_key,
                )
                self._stats.admission_time_ms += (perf_counter_ns() - started) / 1_000_000.0
                return

            free_slots = np.flatnonzero(~self._active)
            if free_slots.size:
                slot = int(free_slots[0])
                self._commit_slot(slot, candidate_key, request, response, vector)
                self._stats.admissions += 1
                self._last_decision = SAGEDecision(
                    admitted=True,
                    reason="free_slot",
                    candidate_key=candidate_key,
                    victim_slot=slot,
                )
                self._stats.admission_time_ms += (perf_counter_ns() - started) / 1_000_000.0
                return

            ghost_vectors = self._ghost.active_vectors()
            candidate_covers = self._space.covers(
                ghost_vectors,
                vector,
                self.config.hit_distance_threshold,
            )
            coverage = self._ghost.active_coverage()
            coverage_counts = self._ghost.active_coverage_counts()
            weights = self._ghost.weights(self._step, self.config.decay_half_life_requests)
            new_gain, victim_losses, deltas = self._scorer.score_all_victims(
                coverage=coverage,
                coverage_counts=coverage_counts,
                candidate_covers=candidate_covers,
                weights=weights,
                resident_active=self._active,
            )
            victim_slot = self._choose_victim(deltas)
            best_delta = float(deltas[victim_slot])
            victim_loss = float(victim_losses[victim_slot])
            victim_key = self._keys[victim_slot]

            if best_delta <= self.config.admission_margin + 1e-12:
                self._stats.rejections += 1
                self._last_decision = SAGEDecision(
                    admitted=False,
                    reason="non_positive_gain",
                    candidate_key=candidate_key,
                    victim_key=victim_key,
                    victim_slot=victim_slot,
                    candidate_new_gain=new_gain,
                    victim_unrecovered_loss=victim_loss,
                    net_delta=best_delta,
                )
            else:
                self._commit_slot(victim_slot, candidate_key, request, response, vector)
                self._stats.admissions += 1
                self._stats.evictions += 1
                self._last_decision = SAGEDecision(
                    admitted=True,
                    reason="positive_replacement_gain",
                    candidate_key=candidate_key,
                    victim_key=victim_key,
                    victim_slot=victim_slot,
                    candidate_new_gain=new_gain,
                    victim_unrecovered_loss=victim_loss,
                    net_delta=best_delta,
                )

        self._stats.admission_time_ms += (perf_counter_ns() - started) / 1_000_000.0

    @property
    def current_size(self) -> int:
        with self._lock:
            return int(np.count_nonzero(self._active))

    @property
    def last_decision(self) -> SAGEDecision | None:
        with self._lock:
            return self._last_decision.model_copy(deep=True) if self._last_decision else None

    def stats(self) -> SAGEStats:
        with self._lock:
            return self._stats.model_copy(deep=True)

    def current_coverage_score(self) -> float:
        with self._lock:
            counts = self._ghost.active_coverage_counts()
            weights = self._ghost.weights(self._step, self.config.decay_half_life_requests)
            return float(weights[counts > 0].sum())

    def resident_snapshot(self) -> list[PersistedResident]:
        with self._lock:
            snapshot: list[PersistedResident] = []
            for slot in np.flatnonzero(self._active):
                snapshot.append(self._resident_model(int(slot)))
            return snapshot

    def reset(self, *, clear_storage: bool = True) -> None:
        with self._lock:
            dimension = self._resident_vectors.shape[1] if self._resident_vectors is not None else None
            self._resident_vectors = (
                np.zeros((self.config.max_size, dimension), dtype=np.float32)
                if dimension is not None
                else None
            )
            self._active[:] = False
            self._keys = [None] * self.config.max_size
            self._prompts = [None] * self.config.max_size
            self._responses = [None] * self.config.max_size
            self._inserted_steps[:] = 0
            self._last_access_steps[:] = 0
            self._ghost = GhostWindow(self.config.ghost_capacity, self.config.max_size)
            self._step = 0
            self._stats = SAGEStats()
            self._last_decision = None
            if clear_storage:
                self._storage.clear()
                self._storage_initialized = False

    def close(self) -> None:
        self._storage.close()

    def _ensure_dimension(self, dimension: int) -> None:
        if self._resident_vectors is None:
            self._resident_vectors = np.zeros(
                (self.config.max_size, dimension), dtype=np.float32
            )
            self._ensure_storage_metadata(dimension)
            return
        if self._resident_vectors.shape[1] != dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._resident_vectors.shape[1]} to {dimension}"
            )

    def _resident_distances_and_coverage(
        self, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full(self.config.max_size, np.inf, dtype=np.float32)
        coverage = np.zeros(self.config.max_size, dtype=np.bool_)
        active_slots = np.flatnonzero(self._active)
        if active_slots.size == 0:
            return distances, coverage
        assert self._resident_vectors is not None
        active_distances = self._space.distances(self._resident_vectors[active_slots], vector)
        distances[active_slots] = active_distances
        coverage[active_slots] = active_distances <= self.config.hit_distance_threshold
        return distances, coverage

    def _choose_victim(self, deltas: np.ndarray) -> int:
        active_deltas = np.where(self._active, deltas, -np.inf)
        best_delta = float(np.max(active_deltas))
        tied = np.flatnonzero(self._active & np.isclose(active_deltas, best_delta, atol=1e-12))
        if tied.size == 1:
            return int(tied[0])
        tied_accesses = self._last_access_steps[tied]
        least_recent = tied[tied_accesses == tied_accesses.min()]
        return int(least_recent.min())

    def _commit_slot(
        self,
        slot: int,
        key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
    ) -> None:
        self._ensure_storage_metadata(vector.shape[0])
        assert self._resident_vectors is not None
        ghost_vectors = self._ghost.active_vectors()
        new_column = self._space.covers(
            ghost_vectors,
            vector,
            self.config.hit_distance_threshold,
        )

        old_active = bool(self._active[slot])
        old_vector = self._resident_vectors[slot].copy()
        old_key = self._keys[slot]
        old_prompt = self._prompts[slot]
        old_response = self._responses[slot]
        old_inserted = int(self._inserted_steps[slot])
        old_last_access = int(self._last_access_steps[slot])
        old_column = self._ghost.coverage_column(slot)

        self._resident_vectors[slot] = vector
        self._active[slot] = True
        self._keys[slot] = key
        self._prompts[slot] = prompt
        self._responses[slot] = response
        self._inserted_steps[slot] = self._step
        self._last_access_steps[slot] = self._step
        self._ghost.replace_resident_column(slot, new_column)

        try:
            self._storage.upsert(self._resident_model(slot))
        except Exception:
            self._resident_vectors[slot] = old_vector
            self._active[slot] = old_active
            self._keys[slot] = old_key
            self._prompts[slot] = old_prompt
            self._responses[slot] = old_response
            self._inserted_steps[slot] = old_inserted
            self._last_access_steps[slot] = old_last_access
            self._ghost.replace_resident_column(slot, old_column)
            raise

    def _resident_model(self, slot: int) -> PersistedResident:
        assert self._resident_vectors is not None
        key = self._keys[slot]
        prompt = self._prompts[slot]
        response = self._responses[slot]
        if key is None or prompt is None or response is None:
            raise RuntimeError(f"Resident slot {slot} is incomplete")
        return PersistedResident(
            slot=slot,
            key=key,
            prompt=prompt,
            response=response,
            vector=self._resident_vectors[slot].tolist(),
            inserted_step=int(self._inserted_steps[slot]),
            last_access_step=int(self._last_access_steps[slot]),
        )

    def _restore_residents(self) -> None:
        residents = self._storage.load()
        if not residents:
            return
        dimension = len(residents[0].vector)
        self._ensure_dimension(dimension)
        assert self._resident_vectors is not None
        seen_slots: set[int] = set()
        for resident in residents:
            if resident.slot >= self.config.max_size:
                raise ValueError(
                    f"Persistent resident slot {resident.slot} exceeds max_size={self.config.max_size}"
                )
            if resident.slot in seen_slots:
                raise ValueError(f"Duplicate persistent resident slot {resident.slot}")
            seen_slots.add(resident.slot)
            vector = self._space.prepare(resident.vector)
            if vector.shape[0] != dimension:
                raise ValueError("Persistent resident vectors have inconsistent dimensions")
            slot = resident.slot
            self._resident_vectors[slot] = vector
            self._active[slot] = True
            self._keys[slot] = resident.key
            self._prompts[slot] = resident.prompt
            self._responses[slot] = resident.response
            self._inserted_steps[slot] = resident.inserted_step
            self._last_access_steps[slot] = resident.last_access_step
            self._step = max(self._step, resident.inserted_step, resident.last_access_step)

    def _ensure_storage_metadata(self, dimension: int) -> None:
        if self._storage_initialized:
            return
        self._storage.validate_or_initialize(
            SAGEStorageMetadata(
                distance_method=self.config.distance_method.value,
                hit_distance_threshold=self.config.hit_distance_threshold,
                vector_dimension=dimension,
            )
        )
        self._storage_initialized = True

    @staticmethod
    def _generate_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
