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
    """Windowed SAGE with soft semantic coverage and two demand horizons."""

    def __init__(
        self,
        max_size: int,
        hit_distance_threshold: float,
        prompt_embedder: Callable[[str], Sequence[float]],
        *,
        ranking_distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
        ghost_capacity: int | None = None,
        decay_half_life_requests: float | None = None,
        admission_margin: float = 0.0,
        current_request_weight: float = 0.1,
        window_fraction: float = 0.05,
        soft_coverage: bool = True,
        soft_coverage_power: float = 1.0,
        recent_history_multiplier: float = 4.0,
        recent_history_limit: int = 4096,
        long_history_capacity: int | None = None,
        long_history_multiplier: float = 8.0,
        long_history_limit: int = 8192,
        long_sample_stride: int = 8,
        recent_evidence_weight: float = 0.7,
        long_decay_half_life_requests: float | None = None,
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
            current_request_weight=current_request_weight,
            window_fraction=window_fraction,
            soft_coverage=soft_coverage,
            soft_coverage_power=soft_coverage_power,
            recent_history_multiplier=recent_history_multiplier,
            recent_history_limit=recent_history_limit,
            long_history_capacity=long_history_capacity,
            long_history_multiplier=long_history_multiplier,
            long_history_limit=long_history_limit,
            long_sample_stride=long_sample_stride,
            recent_evidence_weight=recent_evidence_weight,
            long_decay_half_life_requests=long_decay_half_life_requests,
            storage_path=Path(storage_path) if storage_path is not None else None,
            storage_namespace=storage_namespace,
        )
        super().__init__(config.max_size, "SAGE")
        self.config = config
        self._embedder = prompt_embedder
        self._space = VectorSpace(config.distance_method)
        self._scorer = SAGEScorer()
        self._lock = threading.RLock()
        self._legacy_local = threading.local()

        self._window_mask = np.zeros(config.max_size, dtype=np.bool_)
        self._window_mask[: config.window_size] = True
        self._main_mask = ~self._window_mask
        self._recent = GhostWindow(config.recent_capacity, config.max_size)
        self._long = GhostWindow(config.long_capacity, config.max_size)
        # Backward-compatible private alias used by older diagnostics.
        self._ghost = self._recent
        self._recent_evictions = 0

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
            utilities = self._main_utilities_from_distances(distances)
            observation_id = self._observe(vector, utilities)
            context = SAGELookupContext(
                embedding=vector.tolist(),
                observation_id=observation_id,
                request_step=self._step,
            )

            self._stats.requests += 1
            if bool(coverage.any()):
                slot = int(np.argmin(np.where(coverage, distances, np.inf)))
                response = self._responses[slot]
                if response is None:
                    raise RuntimeError(f"Active resident slot {slot} has no response")
                self._last_access_steps[slot] = self._step
                self._stats.hits += 1
                if self._window_mask[slot]:
                    self._stats.window_hits += 1
                else:
                    self._stats.main_hits += 1
                result = CacheLookup(
                    hit=True,
                    response=response,
                    context=context,
                    metadata={
                        "cache_key": self._keys[slot],
                        "matched_prompt": self._prompts[slot],
                        "prompt_distance": float(distances[slot]),
                        "resident_slot": slot,
                        "segment": "window" if self._window_mask[slot] else "main",
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
                distances, _ = self._resident_distances_and_coverage(vector)
                observation_id = self._observe(
                    vector, self._main_utilities_from_distances(distances)
                )
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

        incoming_key = self._generate_key(request)
        try:
            with self._lock:
                self._ensure_dimension(vector.shape[0])
                _, current_coverage = self._resident_distances_and_coverage(vector)
                if bool(current_coverage.any()):
                    self._stats.rejections += 1
                    self._stats.concurrent_deduplications += 1
                    self._last_decision = SAGEDecision(
                        admitted=False,
                        reason="concurrent_semantic_duplicate",
                        candidate_key=incoming_key,
                    )
                    return
                if self.config.window_size == 0:
                    self._admit_direct(
                        incoming_key, request, response, vector, context.observation_id
                    )
                else:
                    self._admit_through_window(incoming_key, request, response, vector)
        finally:
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
            values = []
            for horizon, half_life, configured_weight in self._horizons():
                if horizon.size == 0:
                    continue
                weights = horizon.weights(self._step, half_life)
                total = float(weights.sum())
                if total > 0.0:
                    values.append(
                        (
                            configured_weight,
                            float(np.sum(weights * horizon.active_top_values()[:, 0]) / total),
                        )
                    )
            factor = sum(item[0] for item in values)
            return sum(weight * score for weight, score in values) / factor if factor else 0.0

    def resident_snapshot(self) -> list[PersistedResident]:
        with self._lock:
            return [self._resident_model(int(slot)) for slot in np.flatnonzero(self._active)]

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
            self._recent = GhostWindow(self.config.recent_capacity, self.config.max_size)
            self._long = GhostWindow(self.config.long_capacity, self.config.max_size)
            self._ghost = self._recent
            self._recent_evictions = 0
            self._step = 0
            self._stats = SAGEStats()
            self._last_decision = None
            if clear_storage:
                self._storage.clear()
                self._storage_initialized = False

    def close(self) -> None:
        self._storage.close()

    def _admit_direct(
        self,
        key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
        observation_id: int,
    ) -> None:
        free = np.flatnonzero(self._main_mask & ~self._active)
        if free.size:
            slot = int(free[0])
            self._commit_batch([(slot, key, prompt, response, vector)])
            self._stats.admissions += 1
            self._last_decision = SAGEDecision(
                admitted=True, reason="free_slot", candidate_key=key, victim_slot=slot
            )
            return

        new_gain, losses, deltas = self._score_candidate(
            vector, current_observation_id=observation_id
        )
        victim = self._choose_victim(deltas)
        best_delta = float(deltas[victim])
        victim_key = self._keys[victim]
        if best_delta <= self.config.admission_margin + 1e-12:
            self._stats.rejections += 1
            self._last_decision = SAGEDecision(
                admitted=False,
                reason="non_positive_gain",
                candidate_key=key,
                victim_key=victim_key,
                victim_slot=victim,
                candidate_new_gain=new_gain,
                victim_unrecovered_loss=float(losses[victim]),
                net_delta=best_delta,
            )
            return
        self._commit_batch([(victim, key, prompt, response, vector)])
        self._stats.admissions += 1
        self._stats.evictions += 1
        self._last_decision = SAGEDecision(
            admitted=True,
            reason="positive_replacement_gain",
            candidate_key=key,
            victim_key=victim_key,
            victim_slot=victim,
            candidate_new_gain=new_gain,
            victim_unrecovered_loss=float(losses[victim]),
            net_delta=best_delta,
        )

    def _admit_through_window(
        self,
        incoming_key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
    ) -> None:
        free_window = np.flatnonzero(self._window_mask & ~self._active)
        if free_window.size:
            slot = int(free_window[0])
            self._commit_batch([(slot, incoming_key, prompt, response, vector)])
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._last_decision = SAGEDecision(
                admitted=True,
                reason="window_insert",
                candidate_key=incoming_key,
                incoming_admitted=True,
                victim_slot=slot,
            )
            return

        window_slots = np.flatnonzero(self._window_mask & self._active)
        outgoing_slot = int(
            window_slots[np.argmin(self._last_access_steps[window_slots])]
        )
        outgoing_key = self._keys[outgoing_slot]
        outgoing_prompt = self._prompts[outgoing_slot]
        outgoing_response = self._responses[outgoing_slot]
        assert self._resident_vectors is not None
        outgoing_vector = self._resident_vectors[outgoing_slot].copy()
        if outgoing_key is None or outgoing_prompt is None or outgoing_response is None:
            raise RuntimeError("Probation window resident is incomplete")

        incoming_update = (outgoing_slot, incoming_key, prompt, response, vector)
        if self.config.main_size == 0:
            self._commit_batch([incoming_update])
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.rejections += 1
            self._stats.promotion_rejections += 1
            self._stats.evictions += 1
            self._last_decision = SAGEDecision(
                admitted=False,
                promoted=False,
                reason="window_candidate_rejected",
                candidate_key=outgoing_key,
                incoming_key=incoming_key,
                incoming_admitted=True,
            )
            return

        free_main = np.flatnonzero(self._main_mask & ~self._active)
        if free_main.size:
            main_slot = int(free_main[0])
            self._commit_batch(
                [
                    (main_slot, outgoing_key, outgoing_prompt, outgoing_response, outgoing_vector),
                    incoming_update,
                ]
            )
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.promotions += 1
            self._last_decision = SAGEDecision(
                admitted=True,
                promoted=True,
                reason="main_free_slot",
                candidate_key=outgoing_key,
                incoming_key=incoming_key,
                incoming_admitted=True,
                victim_slot=main_slot,
            )
            return

        main_slots = np.flatnonzero(self._main_mask & self._active)
        main_distances = self._space.distances(
            self._resident_vectors[main_slots], outgoing_vector
        )
        if bool((main_distances <= self.config.hit_distance_threshold).any()):
            self._commit_batch([incoming_update])
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.rejections += 1
            self._stats.promotion_rejections += 1
            self._stats.evictions += 1
            self._last_decision = SAGEDecision(
                admitted=False,
                promoted=False,
                reason="window_candidate_rejected",
                candidate_key=outgoing_key,
                incoming_key=incoming_key,
                incoming_admitted=True,
            )
            return

        new_gain, losses, deltas = self._score_candidate(outgoing_vector)
        victim = self._choose_victim(deltas)
        best_delta = float(deltas[victim])
        victim_key = self._keys[victim]
        if best_delta <= self.config.admission_margin + 1e-12:
            self._commit_batch([incoming_update])
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.rejections += 1
            self._stats.promotion_rejections += 1
            self._stats.evictions += 1
            self._last_decision = SAGEDecision(
                admitted=False,
                promoted=False,
                reason="window_candidate_rejected",
                candidate_key=outgoing_key,
                incoming_key=incoming_key,
                incoming_admitted=True,
                victim_key=victim_key,
                victim_slot=victim,
                candidate_new_gain=new_gain,
                victim_unrecovered_loss=float(losses[victim]),
                net_delta=best_delta,
            )
            return

        self._commit_batch(
            [
                (victim, outgoing_key, outgoing_prompt, outgoing_response, outgoing_vector),
                incoming_update,
            ]
        )
        self._stats.admissions += 1
        self._stats.window_insertions += 1
        self._stats.promotions += 1
        self._stats.evictions += 1
        self._last_decision = SAGEDecision(
            admitted=True,
            promoted=True,
            reason="window_candidate_promoted",
            candidate_key=outgoing_key,
            incoming_key=incoming_key,
            incoming_admitted=True,
            victim_key=victim_key,
            victim_slot=victim,
            candidate_new_gain=new_gain,
            victim_unrecovered_loss=float(losses[victim]),
            net_delta=best_delta,
        )

    def _score_candidate(
        self,
        vector: np.ndarray,
        *,
        current_observation_id: int | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        resident_active = self._active & self._main_mask
        parts: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for horizon, half_life, configured_weight in self._horizons():
            if horizon.size == 0 or configured_weight == 0.0:
                continue
            candidate = self._utilities_for_vectors(horizon.active_vectors(), vector)
            weights = horizon.weights(
                self._step,
                half_life,
                current_observation_id=current_observation_id,
                current_observation_weight=self.config.current_request_weight,
            )
            total_weight = float(weights.sum())
            if total_weight == 0.0:
                continue
            gain, losses, deltas = self._scorer.score_from_top_two(
                top_values=horizon.active_top_values(),
                top_slots=horizon.active_top_slots(),
                candidate_utilities=candidate,
                weights=weights,
                resident_active=resident_active,
            )
            parts.append((configured_weight, gain / total_weight, losses / total_weight, deltas / total_weight))

        if not parts:
            losses = np.full(self.config.max_size, np.inf, dtype=np.float64)
            deltas = np.full(self.config.max_size, -np.inf, dtype=np.float64)
            deltas[resident_active] = 0.0
            losses[resident_active] = 0.0
            return 0.0, losses, deltas
        factor = sum(part[0] for part in parts)
        new_gain = sum(part[0] * part[1] for part in parts) / factor
        losses = sum((part[0] * part[2] for part in parts), np.zeros(self.config.max_size)) / factor
        deltas = sum((part[0] * part[3] for part in parts), np.zeros(self.config.max_size)) / factor
        deltas[~resident_active] = -np.inf
        losses[~resident_active] = np.inf
        return float(new_gain), losses, deltas

    def _choose_victim(self, deltas: np.ndarray) -> int:
        resident_active = self._active & self._main_mask
        active_deltas = np.where(resident_active, deltas, -np.inf)
        best_delta = float(np.max(active_deltas))
        tied = np.flatnonzero(
            resident_active
            & np.isclose(active_deltas, best_delta, rtol=0.0, atol=1e-12)
        )
        if tied.size == 1:
            return int(tied[0])

        evidence_counts = self._recent.nonzero_counts(tied) + self._long.nonzero_counts(tied)
        zero_evidence = tied[evidence_counts == 0]
        if zero_evidence.size:
            accesses = self._last_access_steps[zero_evidence]
            least_recent = zero_evidence[accesses == accesses.min()]
            return int(least_recent.min())

        responsibilities = self._fractional_responsibilities(tied)
        minimum = float(responsibilities.min())
        tied = tied[np.isclose(responsibilities, minimum, rtol=0.0, atol=1e-12)]
        if tied.size == 1:
            return int(tied[0])
        accesses = self._last_access_steps[tied]
        least_recent = tied[accesses == accesses.min()]
        return int(least_recent.min())

    def _fractional_responsibilities(self, slots: np.ndarray) -> np.ndarray:
        parts: list[tuple[float, np.ndarray]] = []
        for horizon, half_life, configured_weight in self._horizons():
            if horizon.size == 0 or configured_weight == 0.0:
                continue
            weights = horizon.weights(self._step, half_life)
            total = float(weights.sum())
            if total > 0.0:
                parts.append(
                    (
                        configured_weight,
                        horizon.fractional_responsibility(weights, slots) / total,
                    )
                )
        factor = sum(item[0] for item in parts)
        if factor == 0.0:
            return np.zeros(slots.size, dtype=np.float64)
        return sum((weight * values for weight, values in parts), np.zeros(slots.size)) / factor

    def _horizons(self) -> list[tuple[GhostWindow, float | None, float]]:
        return [
            (
                self._recent,
                self.config.decay_half_life_requests,
                self.config.recent_evidence_weight,
            ),
            (
                self._long,
                self.config.long_decay_half_life_requests,
                1.0 - self.config.recent_evidence_weight,
            ),
        ]

    def _observe(self, vector: np.ndarray, utilities: np.ndarray) -> int:
        observation_id, evicted = self._recent.add_with_evicted(
            vector, self._step, utilities
        )
        if evicted is not None:
            self._recent_evictions += 1
            if self._recent_evictions % self.config.long_sample_stride == 0:
                evicted_vector, evicted_step, evicted_utilities = evicted
                self._long.add(evicted_vector, evicted_step, evicted_utilities)
        return observation_id

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

    def _main_utilities_from_distances(self, distances: np.ndarray) -> np.ndarray:
        utilities = np.zeros(self.config.max_size, dtype=np.float32)
        active_main = np.flatnonzero(self._active & self._main_mask)
        if active_main.size:
            utilities[active_main] = self._distance_utilities(distances[active_main])
        return utilities

    def _utilities_for_vectors(self, vectors: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if vectors.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        return self._distance_utilities(self._space.distances(vectors, vector))

    def _distance_utilities(self, distances: np.ndarray) -> np.ndarray:
        threshold = self.config.hit_distance_threshold
        if not self.config.soft_coverage:
            return (distances <= threshold).astype(np.float32)
        if threshold == 0.0:
            return np.isclose(distances, 0.0, rtol=0.0, atol=1e-7).astype(np.float32)
        utilities = np.clip(1.0 - distances / threshold, 0.0, 1.0)
        if self.config.soft_coverage_power != 1.0:
            utilities = np.power(utilities, self.config.soft_coverage_power)
        utilities[distances > threshold] = 0.0
        return utilities.astype(np.float32, copy=False)

    def _commit_batch(
        self,
        updates: list[tuple[int, str, str, str, np.ndarray]],
    ) -> None:
        slots = [update[0] for update in updates]
        if len(slots) != len(set(slots)):
            raise ValueError("A slot may be updated only once per transaction")
        assert self._resident_vectors is not None
        self._ensure_storage_metadata(self._resident_vectors.shape[1])

        old_state = {
            slot: (
                bool(self._active[slot]),
                self._resident_vectors[slot].copy(),
                self._keys[slot],
                self._prompts[slot],
                self._responses[slot],
                int(self._inserted_steps[slot]),
                int(self._last_access_steps[slot]),
                self._recent.utility_column(slot),
                self._long.utility_column(slot),
            )
            for slot in slots
        }
        try:
            for slot, key, prompt, response, vector in updates:
                self._resident_vectors[slot] = vector
                self._active[slot] = True
                self._keys[slot] = key
                self._prompts[slot] = prompt
                self._responses[slot] = response
                self._inserted_steps[slot] = self._step
                self._last_access_steps[slot] = self._step
                if self._main_mask[slot]:
                    recent_column = self._utilities_for_vectors(
                        self._recent.active_vectors(), vector
                    )
                    long_column = self._utilities_for_vectors(
                        self._long.active_vectors(), vector
                    )
                else:
                    recent_column = np.zeros(self._recent.size, dtype=np.float32)
                    long_column = np.zeros(self._long.size, dtype=np.float32)
                self._recent.replace_resident_column(slot, recent_column)
                self._long.replace_resident_column(slot, long_column)
            self._storage.apply([self._resident_model(slot) for slot in slots])
        except Exception:
            for slot in slots:
                (
                    active,
                    resident_vector,
                    key,
                    prompt,
                    response,
                    inserted,
                    last_access,
                    recent_column,
                    long_column,
                ) = old_state[slot]
                self._active[slot] = active
                self._resident_vectors[slot] = resident_vector
                self._keys[slot] = key
                self._prompts[slot] = prompt
                self._responses[slot] = response
                self._inserted_steps[slot] = inserted
                self._last_access_steps[slot] = last_access
                self._recent.replace_resident_column(slot, recent_column)
                self._long.replace_resident_column(slot, long_column)
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
                window_size=self.config.window_size,
                soft_coverage=self.config.soft_coverage,
                soft_coverage_power=self.config.soft_coverage_power,
            )
        )
        self._storage_initialized = True

    @staticmethod
    def _generate_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
