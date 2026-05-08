"""
policy.py — Semantic Resonance Cache (SRC) policy.

Implements ``SRCSimilarityCache``, an EchoLLM-compatible cache policy that
inherits from ``SimilarityCache`` and adds LLM-aware admission and eviction.

Public surface
--------------
    SRCSimilarityCache   — full cache class (inherit / use directly)
    SRCPolicy            — alias for experiments
"""

from __future__ import annotations

import logging
import random
from typing import Callable

from cache.similarity_cache.ranking_distance_method import RankingDistanceMethod
from cache.similarity_cache.similarity_cache import SimilarityCache
from cache.storage_client.faiss_client import FaissDistanceMethod
from cache.storage_client.records import EmbeddedRequestRecord, ResponseRecord
from .models import GhostHistory, SRCItemMeta
from .safety import (
    has_private_pattern,
    has_time_sensitive_phrase,
    safety_score,
)
from .scoring import (
    compute_cost_hat,
    compute_demand,
    estimate_tokens,
    normalize_demand,
    normalize_embedding,
    src_score,
)

logger = logging.getLogger("EchoLLM.SRC")

# ---------------------------------------------------------------------------
# Default hyper-parameters (spec §5)
# ---------------------------------------------------------------------------

_DEFAULT_K: int = 8  # neighbours for demand estimation
_DEFAULT_S: int = 16  # eviction sample size
_DEFAULT_THETA_HIT: float = 0.90  # minimum cosine sim for a semantic hit
_DEFAULT_THETA_NEAR: float = 0.80  # minimum cosine sim to count as "nearby"
_DEFAULT_THETA_SAFE: float = 0.50  # minimum safety score to admit/reuse
_DEFAULT_EPSILON: float = 0.05  # anti-churn margin for eviction


class SRCSimilarityCache(SimilarityCache):
    """
    Semantic Resonance Cache — LLM-aware admission and eviction policy.

    Design overview
    ---------------
    SRC augments EchoLLM's ``SimilarityCache`` with three scoring axes:

    * **Cost (Ĉ)**   — how expensive was it to generate this response?
    * **Safety (R)** — is this response safe and stable to reuse?
    * **Demand (D̂)** — does this prompt belong to a dense semantic region?

    The unified score ``S = D̂ · Ĉ · R`` drives both admission (must beat the
    weakest cached item by a margin ``ε``) and eviction (lowest-scoring sampled
    item is displaced).

    A ghost history tracks rejected / evicted semantic regions so that demand
    estimation improves even after eviction.

    Parameters
    ----------
    max_size : int
        Cache capacity (maximum number of entries B).
    prompt_embedder : Callable[[str], list[float]]
        EchoLLM embedding function.
    ranking_distance_method : RankingDistanceMethod
        Re-ranking metric used by EchoLLM's RequestsDB.
    db_distance_method : FaissDistanceMethod
        Distance metric used by the FAISS vector index.
    candidates_number : int
        Number of FAISS nearest-neighbour candidates fetched per lookup
        (EchoLLM's ``candidates_number``).
    k : int
        Neighbourhood size for semantic-demand estimation.
    s : int
        Number of randomly sampled eviction candidates.
    theta_hit : float
        Minimum cosine similarity threshold for a semantic cache hit.
    theta_near : float
        Minimum cosine similarity for a neighbour to contribute to demand.
    theta_safe : float
        Minimum safety score required to admit or reuse a cached item.
    epsilon : float
        Anti-churn margin: new item must score > (1 + ε) × victim score.
    seed : int | None
        Random seed for reproducible sampled eviction.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
            self,
            max_size: int,
            prompt_embedder: Callable[[str], list[float]],
            ranking_distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
            db_distance_method: FaissDistanceMethod = FaissDistanceMethod.COSINE,
            candidates_number: int = 100,
            # SRC hyper-parameters
            k: int = _DEFAULT_K,
            s: int = _DEFAULT_S,
            theta_hit: float = _DEFAULT_THETA_HIT,
            theta_near: float = _DEFAULT_THETA_NEAR,
            theta_safe: float = _DEFAULT_THETA_SAFE,
            epsilon: float = _DEFAULT_EPSILON,
            seed: int | None = 42,
    ) -> None:
        # EchoLLM's SimilarityCache uses a *distance* threshold, while SRC
        # works with cosine *similarity*.  Convert:  distance = 1 − similarity.
        hit_distance_threshold = 1.0 - theta_hit

        super().__init__(
            max_size=max_size,
            hit_distance_threshold=hit_distance_threshold,
            candidates_number=candidates_number,
            ranking_distance_method=ranking_distance_method,
            db_distance_method=db_distance_method,
            prompt_embedder=prompt_embedder,
            policy_name="Semantic Resonance Cache",
        )

        # SRC hyper-parameters
        self._k = k
        self._s = s
        self._theta_hit = theta_hit
        self._theta_near = theta_near
        self._theta_safe = theta_safe
        self._epsilon = epsilon

        # Per-item SRC metadata: prompt_key → SRCItemMeta
        self._metadata: dict[str, SRCItemMeta] = {}

        # Ghost history
        self._ghost: GhostHistory = GhostHistory(cache_capacity=max_size)

        # Reproducible random sampling for eviction
        self._rng: random.Random = random.Random(seed)

        logger.info(
            "SRCSimilarityCache initialised  B=%d  k=%d  s=%d  "
            "θ_hit=%.2f  θ_near=%.2f  θ_safe=%.2f  ε=%.3f  seed=%s",
            max_size, k, s, theta_hit, theta_near, theta_safe, epsilon, seed,
        )

    # ------------------------------------------------------------------
    # ICache interface — hit path
    # ------------------------------------------------------------------

    def is_hit(self, prompt: str) -> bool:
        """
        Return True iff the prompt can be served safely from cache.

        Extends the base cosine-distance check with SRC's safe-reuse
        conditions (spec §13 "Safe Semantic Reuse").
        """
        # Fast rejection: time-sensitive or private incoming prompt
        if has_time_sensitive_phrase(prompt) or has_private_pattern(prompt):
            return False

        result = self._requests_db.most_similar_request(
            tuple(self._embedder(prompt)),  # hashable for LRU cache inside RequestsDB
            self._candidates_number,
        )
        if result is None:
            return False

        hit_record, distance = result

        # Distance threshold check (distance = 1 − cosine_similarity)
        if distance > self._hit_distance_threshold:
            return False

        # SRC safety check on the cached item
        meta = self._metadata.get(hit_record.key)
        if meta is None:
            # Item was inserted before SRC tracked it (e.g. external insertion)
            return False

        return meta.safety >= self._theta_safe

    def on_hit(self, prompt: str, **kwargs) -> str:
        """Return the cached response for *prompt* (guaranteed hit path)."""
        hit_record, _ = self._requests_db.most_similar_request(
            tuple(self._embedder(prompt)),
            self._candidates_number,
        )
        response = self._responses_db.fetch_by_request(hit_record.key)
        if response is None:
            raise KeyError(
                f"Response for request_key=`{hit_record.key}` not found in ResponsesDB"
            )
        return response.response

    # ------------------------------------------------------------------
    # ICache interface — miss path
    # ------------------------------------------------------------------

    def on_miss(
            self,
            prompt: str,
            llm_response: str,
            llm_latency: float = 0.0,
            **kwargs,
    ) -> None:
        """
        Process a cache miss: score the candidate and decide admission.

        Parameters
        ----------
        prompt : str
            The user prompt (not yet cached).
        llm_response : str
            The response just generated by the real LLM backend.
        llm_latency : float
            LLM generation latency in milliseconds (from EchoLLM's backend).
        **kwargs
            Any additional keyword arguments passed through from EchoLLM.
        """
        # ----------------------------------------------------------------
        # 1. Compute candidate scores
        # ----------------------------------------------------------------
        prompt_key = self._generate_key(prompt)
        embedding_raw = self._embedder(prompt)
        embedding_norm = normalize_embedding(embedding_raw)

        tokens = estimate_tokens(prompt, llm_response)
        cost_hat = compute_cost_hat(latency_ms=llm_latency, token_count=tokens)
        r_score = safety_score(prompt, llm_response)

        demand_raw = self._compute_demand_for_prompt(
            embedding_raw=embedding_raw,
            exclude_key=None,  # new item not in cache yet
        )
        demand_hat = normalize_demand(demand_raw, self._k)
        s_new = src_score(demand_hat, cost_hat, r_score)

        logger.debug(
            "Miss  key=%s  Ĉ=%.4f  R=%.2f  D̂=%.4f  S=%.4f",
            prompt_key[:8], cost_hat, r_score, demand_hat, s_new,
        )

        # ----------------------------------------------------------------
        # 2. Reject if unsafe
        # ----------------------------------------------------------------
        if r_score < self._theta_safe:
            logger.debug("Rejected (unsafe R=%.2f)  key=%s", r_score, prompt_key[:8])
            self._ghost.add(embedding_norm, reason="rejected")
            return

        # ----------------------------------------------------------------
        # 3. Admit if space is available
        # ----------------------------------------------------------------
        if self.current_size() < self._max_size:
            self._admit(
                prompt=prompt,
                prompt_key=prompt_key,
                embedding_raw=embedding_raw,
                llm_response=llm_response,
                cost_hat=cost_hat,
                r_score=r_score,
            )
            return

        # ----------------------------------------------------------------
        # 4. Cache is full — sampled eviction
        # ----------------------------------------------------------------
        victim_key, victim_score = self._find_victim()

        if victim_key is None:
            # Cache is empty (shouldn't happen but be safe)
            self._admit(
                prompt=prompt,
                prompt_key=prompt_key,
                embedding_raw=embedding_raw,
                llm_response=llm_response,
                cost_hat=cost_hat,
                r_score=r_score,
            )
            return

        if s_new > (1.0 + self._epsilon) * victim_score:
            logger.debug(
                "Evicting victim key=%s  S_victim=%.4f  S_new=%.4f",
                victim_key[:8], victim_score, s_new,
            )
            self._evict(victim_key)
            self._admit(
                prompt=prompt,
                prompt_key=prompt_key,
                embedding_raw=embedding_raw,
                llm_response=llm_response,
                cost_hat=cost_hat,
                r_score=r_score,
            )
        else:
            logger.debug(
                "Rejected (score too low)  S_new=%.4f  S_victim=%.4f  key=%s",
                s_new, victim_score, prompt_key[:8],
            )
            self._ghost.add(embedding_norm, reason="rejected")

    # ------------------------------------------------------------------
    # current_size (override to satisfy both ICache.current_size and
    # the property / method dual usage pattern in EchoLLM)
    # ------------------------------------------------------------------

    def current_size(self) -> int:  # type: ignore[override]
        return self._responses_db.size()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _admit(
            self,
            prompt: str,
            prompt_key: str,
            embedding_raw: list[float],
            llm_response: str,
            cost_hat: float,
            r_score: float,
    ) -> None:
        """Write the item into EchoLLM's DBs and record SRC metadata."""
        # Save embedding into FAISS + SQLite via EchoLLM's infrastructure
        self._requests_db.save(
            EmbeddedRequestRecord(key=prompt_key, vector=embedding_raw)
        )
        response_key = self._generate_key(llm_response)
        self._responses_db.save(
            ResponseRecord(
                key=response_key,
                request_key=prompt_key,
                response=llm_response,
            )
        )
        # Store SRC decision metadata (no duplication of prompt/response data)
        self._metadata[prompt_key] = SRCItemMeta(
            cost_hat=cost_hat,
            safety=r_score,
        )
        logger.debug("Admitted  key=%s  Ĉ=%.4f  R=%.2f", prompt_key[:8], cost_hat, r_score)

    def _evict(self, victim_key: str) -> None:
        """Remove *victim_key* from EchoLLM's DBs, SRC metadata, and ghost."""
        # Retrieve victim embedding before deletion (for ghost insertion)
        victim_embedding: tuple[float, ...] | None = None
        faiss_items = self._requests_db._faiss_client._items
        if victim_key in faiss_items:
            raw_vec = self._requests_db._faiss_client._reconstruct_original_vector(
                faiss_items[victim_key]
            )
            victim_embedding = normalize_embedding(raw_vec)

        # Remove from EchoLLM's storage
        self._requests_db.remove(victim_key)
        self._responses_db.remove_by_request(victim_key)

        # Remove SRC metadata
        self._metadata.pop(victim_key, None)

        # Add ghost entry for the evicted semantic region
        if victim_embedding is not None:
            self._ghost.add(victim_embedding, reason="evicted")

    def _find_victim(self) -> tuple[str | None, float]:
        """
        Sample *s* cached items and return the one with the lowest SRC score.

        Returns
        -------
        (victim_key, victim_score) or (None, 0.0) if the cache is empty.
        """
        all_keys = list(self._metadata.keys())
        if not all_keys:
            return None, 0.0

        # Sample min(s, cache_size) keys without replacement
        sample_size = min(self._s, len(all_keys))
        sampled_keys = self._rng.sample(all_keys, k=sample_size)

        best_victim_key: str | None = None
        best_victim_score: float = float("inf")

        faiss_items = self._requests_db._faiss_client._items

        for key in sampled_keys:
            meta = self._metadata.get(key)
            if meta is None:
                continue

            # Retrieve raw embedding for demand computation
            if key in faiss_items:
                raw_vec = self._requests_db._faiss_client._reconstruct_original_vector(
                    faiss_items[key]
                )
            else:
                # Embedding not available — assign minimal score so it is evicted
                score = 0.0
                if score < best_victim_score:
                    best_victim_score = score
                    best_victim_key = key
                continue

            demand_raw = self._compute_demand_for_prompt(
                embedding_raw=list(raw_vec),
                exclude_key=key,  # don't count self as a neighbour
            )
            demand_hat = normalize_demand(demand_raw, self._k)
            score = src_score(demand_hat, meta.cost_hat, meta.safety)

            if score < best_victim_score:
                best_victim_score = score
                best_victim_key = key

        return best_victim_key, best_victim_score

    def _compute_demand_for_prompt(
            self,
            embedding_raw: list[float],
            exclude_key: str | None,
    ) -> float:
        """
        Compute raw semantic demand D(p) for a prompt embedding.

        Fetches up to *k* nearest neighbours from the FAISS index (cache)
        and compares against up to *k* ghost entries.

        Parameters
        ----------
        embedding_raw : list[float]
            Raw (unnormalised) prompt embedding.
        exclude_key : str | None
            If given, skip this key when counting cache neighbours (used
            when computing demand for an item that is already in cache).
        """
        # --- Cache neighbours ---
        # Fetch k+1 to allow filtering out self
        fetch_k = self._k + (1 if exclude_key is not None else 0)
        cache_neighbors_raw = self._requests_db._faiss_client.fetch_nearest_k(
            embedding_raw, k=max(fetch_k, 1)
        )

        cache_embeddings: list[tuple[float, ...]] = []
        for stored_vec in cache_neighbors_raw:
            if stored_vec.key == exclude_key:
                continue
            cache_embeddings.append(tuple(stored_vec.vector))
            if len(cache_embeddings) >= self._k:
                break

        # --- Ghost neighbours ---
        # Sample up to k ghost entries for efficiency
        ghost_entries = self._ghost.entries()
        if len(ghost_entries) > self._k:
            ghost_entries = self._rng.sample(ghost_entries, self._k)

        ghost_embeddings: list[tuple[float, ...]] = [
            g.embedding for g in ghost_entries
        ]

        return compute_demand(
            query_embedding=embedding_raw,
            cache_embeddings=cache_embeddings,
            ghost_embeddings=ghost_embeddings,
            k=self._k,
            theta_near=self._theta_near,
        )

    # ------------------------------------------------------------------
    # Diagnostic helpers (useful for tests and benchmarks)
    # ------------------------------------------------------------------

    def get_metadata(self, prompt_key: str) -> SRCItemMeta | None:
        """Return SRC metadata for *prompt_key*, or None if not cached."""
        return self._metadata.get(prompt_key)

    def ghost_size(self) -> int:
        """Return the current number of ghost history entries."""
        return len(self._ghost)

    def cached_keys(self) -> list[str]:
        """Return a list of all currently cached prompt keys."""
        return list(self._metadata.keys())


SRCPolicy = SRCSimilarityCache
