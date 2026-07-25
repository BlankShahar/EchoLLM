import hashlib
import threading
from collections.abc import Callable, Sequence
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np

from cache import CacheLookup, ICache
from cache.sage.distance import VectorSpace
from cache.similarity_cache import RankingDistanceMethod

from .config import SPARQConfig
from .models import SPARQDecision, SPARQLookupContext, SPARQResident, SPARQStats


class SPARQSimilarityCache(ICache):
    """Windowed semantic-frequency cache with proximity-weighted credit."""

    def __init__(
        self,
        max_size: int,
        hit_distance_threshold: float,
        prompt_embedder: Callable[[str], Sequence[float]],
        *,
        ranking_distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
        window_fraction: float = 0.2,
        credit_power: float = 2.0,
        aging_interval_requests: int | None = None,
        aging_factor: float = 0.5,
        admission_margin: float = 0.0,
        initial_score: float = 1.0,
    ) -> None:
        config = SPARQConfig(
            max_size=max_size,
            hit_distance_threshold=hit_distance_threshold,
            distance_method=ranking_distance_method,
            window_fraction=window_fraction,
            credit_power=credit_power,
            aging_interval_requests=aging_interval_requests,
            aging_factor=aging_factor,
            admission_margin=admission_margin,
            initial_score=initial_score,
        )
        super().__init__(config.max_size, "SPARQ")
        self.config = config
        self._embedder = prompt_embedder
        self._space = VectorSpace(config.distance_method)
        self._lock = threading.RLock()
        self._legacy_local = threading.local()

        self._window_mask = np.zeros(config.max_size, dtype=np.bool_)
        self._window_mask[: config.window_size] = True
        self._main_mask = ~self._window_mask
        self._vectors: np.ndarray | None = None
        self._active = np.zeros(config.max_size, dtype=np.bool_)
        self._keys: list[str | None] = [None] * config.max_size
        self._prompts: list[str | None] = [None] * config.max_size
        self._responses: list[str | None] = [None] * config.max_size
        self._scores = np.zeros(config.max_size, dtype=np.float64)
        self._inserted_steps = np.zeros(config.max_size, dtype=np.int64)
        self._last_access_steps = np.zeros(config.max_size, dtype=np.int64)
        self._step = 0
        self._stats = SPARQStats()
        self._last_decision: SPARQDecision | None = None

    def lookup(self, request: str) -> CacheLookup:
        started = perf_counter_ns()
        vector = self._space.prepare(list(self._embedder(request)))
        with self._lock:
            self._ensure_dimension(vector.shape[0])
            self._step += 1
            self._age_scores_if_due()
            distances, coverage = self._distances_and_coverage(vector)
            self._stats.requests += 1
            context = SPARQLookupContext(embedding=vector.tolist())

            if not bool(coverage.any()):
                self._stats.misses += 1
                result = CacheLookup(hit=False, context=context)
            else:
                slot = int(np.argmin(np.where(coverage, distances, np.inf)))
                self._award_credit(distances, coverage)
                self._last_access_steps[slot] = self._step
                self._stats.hits += 1
                if self._window_mask[slot]:
                    self._stats.window_hits += 1
                else:
                    self._stats.main_hits += 1
                response = self._responses[slot]
                if response is None:
                    raise RuntimeError(f"Active resident slot {slot} has no response")
                result = CacheLookup(
                    hit=True,
                    response=response,
                    context=context,
                    metadata={
                        "cache_key": self._keys[slot],
                        "matched_prompt": self._prompts[slot],
                        "prompt_distance": float(distances[slot]),
                        "resident_slot": slot,
                        "score": float(self._scores[slot]),
                        "segment": self._segment(slot),
                    },
                )

            self._stats.lookup_time_ms += (
                perf_counter_ns() - started
            ) / 1_000_000.0
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
                self._age_scores_if_due()
                self._stats.requests += 1
                self._stats.misses += 1
        else:
            context = SPARQLookupContext.model_validate(raw_context)
            vector = np.asarray(context.embedding, dtype=np.float32)

        incoming_key = self._generate_key(request)
        try:
            with self._lock:
                self._ensure_dimension(vector.shape[0])
                _, coverage = self._distances_and_coverage(vector)
                if bool(coverage.any()):
                    self._stats.rejections += 1
                    self._stats.concurrent_deduplications += 1
                    self._last_decision = SPARQDecision(
                        admitted=False,
                        reason="concurrent_semantic_duplicate",
                        incoming_key=incoming_key,
                    )
                    return

                if self.config.window_size == 0:
                    self._insert_direct(incoming_key, request, response, vector)
                else:
                    self._insert_through_window(
                        incoming_key,
                        request,
                        response,
                        vector,
                    )
        finally:
            self._stats.admission_time_ms += (
                perf_counter_ns() - started
            ) / 1_000_000.0

    @property
    def current_size(self) -> int:
        with self._lock:
            return int(np.count_nonzero(self._active))

    @property
    def last_decision(self) -> SPARQDecision | None:
        with self._lock:
            return (
                self._last_decision.model_copy(deep=True)
                if self._last_decision is not None
                else None
            )

    def stats(self) -> SPARQStats:
        with self._lock:
            return self._stats.model_copy(deep=True)

    def resident_snapshot(self) -> list[SPARQResident]:
        with self._lock:
            return [
                self._resident_model(int(slot))
                for slot in np.flatnonzero(self._active)
            ]

    def reset(self) -> None:
        with self._lock:
            dimension = self._vectors.shape[1] if self._vectors is not None else None
            self._vectors = (
                np.zeros((self.config.max_size, dimension), dtype=np.float32)
                if dimension is not None
                else None
            )
            self._active[:] = False
            self._keys = [None] * self.config.max_size
            self._prompts = [None] * self.config.max_size
            self._responses = [None] * self.config.max_size
            self._scores[:] = 0.0
            self._inserted_steps[:] = 0
            self._last_access_steps[:] = 0
            self._step = 0
            self._stats = SPARQStats()
            self._last_decision = None

    def _insert_direct(
        self,
        key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
    ) -> None:
        free = np.flatnonzero(self._main_mask & ~self._active)
        if free.size:
            slot = int(free[0])
            self._write_slot(slot, key, prompt, response, vector)
            self._stats.admissions += 1
            self._last_decision = SPARQDecision(
                admitted=True,
                reason="direct_insert",
                incoming_key=key,
                candidate_key=key,
                victim_slot=slot,
                candidate_score=self.config.initial_score,
                incoming_admitted=True,
            )
            return

        victim = self._main_victim()
        victim_score = float(self._scores[victim])
        candidate_score = self.config.initial_score
        victim_key = self._keys[victim]
        if candidate_score + 1e-12 < victim_score + self.config.admission_margin:
            self._stats.rejections += 1
            self._last_decision = SPARQDecision(
                admitted=False,
                reason="direct_reject",
                incoming_key=key,
                candidate_key=key,
                victim_key=victim_key,
                victim_slot=victim,
                candidate_score=candidate_score,
                victim_score=victim_score,
            )
            return

        self._write_slot(victim, key, prompt, response, vector)
        self._stats.admissions += 1
        self._stats.evictions += 1
        self._last_decision = SPARQDecision(
            admitted=True,
            reason="direct_replace",
            incoming_key=key,
            candidate_key=key,
            victim_key=victim_key,
            victim_slot=victim,
            candidate_score=candidate_score,
            victim_score=victim_score,
            incoming_admitted=True,
        )

    def _insert_through_window(
        self,
        key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
    ) -> None:
        free_window = np.flatnonzero(self._window_mask & ~self._active)
        if free_window.size:
            slot = int(free_window[0])
            self._write_slot(slot, key, prompt, response, vector)
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._last_decision = SPARQDecision(
                admitted=True,
                reason="window_insert",
                incoming_key=key,
                candidate_key=key,
                victim_slot=slot,
                candidate_score=self.config.initial_score,
                incoming_admitted=True,
            )
            return

        window_slots = np.flatnonzero(self._window_mask & self._active)
        outgoing = int(
            window_slots[np.argmin(self._last_access_steps[window_slots])]
        )
        outgoing_key = self._keys[outgoing]
        outgoing_score = float(self._scores[outgoing])
        if outgoing_key is None:
            raise RuntimeError("Active probation slot has no key")

        if self.config.main_size == 0:
            self._write_slot(outgoing, key, prompt, response, vector)
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.evictions += 1
            self._stats.rejections += 1
            self._stats.promotion_rejections += 1
            self._last_decision = SPARQDecision(
                admitted=False,
                reason="window_replace",
                incoming_key=key,
                candidate_key=outgoing_key,
                candidate_score=outgoing_score,
                incoming_admitted=True,
            )
            return

        free_main = np.flatnonzero(self._main_mask & ~self._active)
        if free_main.size:
            main_slot = int(free_main[0])
            self._copy_slot(outgoing, main_slot)
            self._write_slot(outgoing, key, prompt, response, vector)
            self._stats.admissions += 1
            self._stats.window_insertions += 1
            self._stats.promotions += 1
            self._last_decision = SPARQDecision(
                admitted=True,
                reason="main_free_slot",
                incoming_key=key,
                candidate_key=outgoing_key,
                victim_slot=main_slot,
                candidate_score=outgoing_score,
                incoming_admitted=True,
                promoted=True,
            )
            return

        victim = self._main_victim()
        victim_key = self._keys[victim]
        victim_score = float(self._scores[victim])
        promote = (
            outgoing_score
            > victim_score + self.config.admission_margin + 1e-12
        )
        if promote:
            self._copy_slot(outgoing, victim)
            reason = "candidate_promoted"
            self._stats.promotions += 1
        else:
            reason = "candidate_rejected"
            self._stats.rejections += 1
            self._stats.promotion_rejections += 1

        self._write_slot(outgoing, key, prompt, response, vector)
        self._stats.admissions += 1
        self._stats.window_insertions += 1
        self._stats.evictions += 1
        self._last_decision = SPARQDecision(
            admitted=promote,
            reason=reason,
            incoming_key=key,
            candidate_key=outgoing_key,
            victim_key=victim_key,
            victim_slot=victim,
            candidate_score=outgoing_score,
            victim_score=victim_score,
            incoming_admitted=True,
            promoted=promote,
        )

    def _award_credit(
        self,
        distances: np.ndarray,
        coverage: np.ndarray,
    ) -> None:
        slots = np.flatnonzero(coverage)
        if slots.size == 0:
            return
        threshold = self.config.hit_distance_threshold
        if threshold == 0.0:
            closeness = np.ones(slots.size, dtype=np.float64)
        else:
            closeness = np.clip(
                1.0 - distances[slots].astype(np.float64) / threshold,
                0.0,
                1.0,
            )
            if self.config.credit_power != 1.0:
                closeness = np.power(closeness, self.config.credit_power)
        total = float(closeness.sum())
        if total <= 0.0:
            return

        # A request contributes at most one point. Borderline hits contribute
        # less, while overlapping residents split the same semantic demand.
        credit_mass = float(closeness.max())
        self._scores[slots] += credit_mass * closeness / total

    def _age_scores_if_due(self) -> None:
        if self._step % self.config.effective_aging_interval != 0:
            return
        self._scores[self._active] *= self.config.aging_factor
        self._stats.aging_events += 1

    def _main_victim(self) -> int:
        slots = np.flatnonzero(self._main_mask & self._active)
        if slots.size == 0:
            raise RuntimeError("No active main-cache resident is available")
        minimum = float(self._scores[slots].min())
        tied = slots[
            np.isclose(self._scores[slots], minimum, rtol=0.0, atol=1e-12)
        ]
        accesses = self._last_access_steps[tied]
        least_recent = tied[accesses == accesses.min()]
        return int(least_recent.min())

    def _distances_and_coverage(
        self,
        vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full(self.config.max_size, np.inf, dtype=np.float32)
        active_slots = np.flatnonzero(self._active)
        if active_slots.size:
            assert self._vectors is not None
            distances[active_slots] = self._space.distances(
                self._vectors[active_slots],
                vector,
            )
        return (
            distances,
            self._active
            & (distances <= self.config.hit_distance_threshold),
        )

    def _write_slot(
        self,
        slot: int,
        key: str,
        prompt: str,
        response: str,
        vector: np.ndarray,
    ) -> None:
        assert self._vectors is not None
        self._vectors[slot] = vector
        self._active[slot] = True
        self._keys[slot] = key
        self._prompts[slot] = prompt
        self._responses[slot] = response
        self._scores[slot] = self.config.initial_score
        self._inserted_steps[slot] = self._step
        self._last_access_steps[slot] = self._step

    def _copy_slot(self, source: int, target: int) -> None:
        assert self._vectors is not None
        self._vectors[target] = self._vectors[source]
        self._active[target] = True
        self._keys[target] = self._keys[source]
        self._prompts[target] = self._prompts[source]
        self._responses[target] = self._responses[source]
        self._scores[target] = self._scores[source]
        self._inserted_steps[target] = self._inserted_steps[source]
        self._last_access_steps[target] = self._last_access_steps[source]

    def _resident_model(self, slot: int) -> SPARQResident:
        assert self._vectors is not None
        key = self._keys[slot]
        prompt = self._prompts[slot]
        response = self._responses[slot]
        if key is None or prompt is None or response is None:
            raise RuntimeError(f"Active resident slot {slot} is incomplete")
        return SPARQResident(
            slot=slot,
            segment=self._segment(slot),
            key=key,
            prompt=prompt,
            response=response,
            vector=self._vectors[slot].tolist(),
            score=float(self._scores[slot]),
            inserted_step=int(self._inserted_steps[slot]),
            last_access_step=int(self._last_access_steps[slot]),
        )

    def _segment(self, slot: int) -> Literal["window", "main"]:
        return "window" if self._window_mask[slot] else "main"

    def _ensure_dimension(self, dimension: int) -> None:
        if self._vectors is None:
            self._vectors = np.zeros(
                (self.config.max_size, dimension),
                dtype=np.float32,
            )
            return
        if self._vectors.shape[1] != dimension:
            raise ValueError(
                f"Embedding dimension changed from {self._vectors.shape[1]} "
                f"to {dimension}"
            )

    @staticmethod
    def _generate_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
