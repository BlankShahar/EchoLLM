import hashlib
import json
import math
import random
import time
from typing import Any, Callable, Optional

from .models import Lookup, SRCConfig, SRCGhost, SRCMeta
from .safety import has_private_marker, has_time_marker, safety_score
from .scoring import approx_tokens, cosine, eviction_value, saved_cost
from ..similarity_cache import SimilarityCache
from ..similarity_cache.ranking_distance_method import RankingDistanceMethod
from ..storage_client.faiss_client import FaissDistanceMethod
from ..storage_client.records import EmbeddedRequestRecord, ResponseRecord


class SRCSimilarityCache(SimilarityCache):
    """Compact EchoLLM-compatible implementation of Semantic Resonance Cache."""

    def __init__(
            self,
            max_size: int,
            hit_distance_threshold: float = 0.10,
            candidates_number: int = 8,
            ranking_distance_method: RankingDistanceMethod = RankingDistanceMethod.COSINE,
            db_distance_method: FaissDistanceMethod = FaissDistanceMethod.COSINE,
            prompt_embedder: Optional[Callable[[str], list[float]]] = None,
            config: Optional[SRCConfig] = None,
            **config_overrides: Any,
    ):
        if prompt_embedder is None:
            raise ValueError("SRC requires prompt_embedder.")

        self.config = config or SRCConfig(**config_overrides)
        if self.config.ghost_max_size is None:
            self.config.ghost_max_size = 2 * max_size

        super().__init__(
            max_size=max_size,
            hit_distance_threshold=hit_distance_threshold,
            candidates_number=max(candidates_number, self.config.k),
            ranking_distance_method=ranking_distance_method,
            db_distance_method=db_distance_method,
            prompt_embedder=prompt_embedder,
            policy_name="Semantic Resonance Cache (SRC)",
        )

        self._rng = random.Random(self.config.seed)
        self._meta: dict[str, SRCMeta] = {}
        self._ghosts: list[SRCGhost] = []
        self._last_lookup: Optional[Lookup] = None
        self.last_event: dict[str, Any] = {}

    def current_size(self) -> int:
        return len(self._meta)

    def is_hit(self, prompt: str, **kwargs: Any) -> bool:
        now = self._now(kwargs)
        namespace = self._namespace(kwargs)
        normalized = self._normalize(prompt)
        self._last_lookup = Lookup(prompt, namespace, normalized)

        if not normalized:
            self.last_event = self._event("miss", now, namespace, reason="empty_prompt")
            return False

        key = self._key(normalized, namespace)
        if key in self._meta and self._usable(self._meta[key], now):
            self._last_lookup = Lookup(prompt, namespace, normalized, hit=True, hit_type="exact", key=key,
                                       similarity=1.0)
            self.last_event = self._event("hit", now, namespace, hit_type="exact", key=key, similarity=1.0)
            return True

        embedding = self._embed(normalized)
        if embedding is None:
            self.last_event = self._event("miss", now, namespace, reason="embedding_failure")
            return False

        best = 0.0
        near: list[str] = []
        for meta, sim in self._neighbors(embedding, namespace):
            best = max(best, sim)
            if sim >= self.config.theta_hit and self._safe_reuse(normalized, namespace, meta, sim, now):
                self._last_lookup = Lookup(prompt, namespace, normalized, embedding, True, "semantic", meta.key, sim)
                self.last_event = self._event("hit", now, namespace, hit_type="semantic", key=meta.key, similarity=sim)
                return True
            if self.config.theta_near <= sim < self.config.theta_hit:
                near.append(meta.key)

        for near_key in near:
            self._meta[near_key].near_misses += 1
            self._meta[near_key].last_access_at = now

        self._last_lookup = Lookup(prompt, namespace, normalized, embedding, near_keys=near, similarity=best)
        self.last_event = self._event("miss", now, namespace, reason="miss", similarity=best, near=len(near))
        return False

    def on_hit(self, prompt: str, **kwargs: Any) -> str:
        lookup = self._last_lookup
        if lookup is None or not lookup.hit or lookup.key is None:
            raise KeyError("SRC on_hit called without a valid hit.")

        meta = self._meta[lookup.key]
        now = self._now(kwargs)
        meta.last_access_at = now

        if lookup.hit_type == "exact":
            meta.exact_hits += 1
        else:
            old = meta.semantic_hits
            meta.semantic_hits += 1
            meta.avg_semantic_similarity = (meta.avg_semantic_similarity * old + lookup.similarity) / meta.semantic_hits

        response = self._responses_db.fetch_by_request(lookup.key).response
        self.last_event = self._event("hit", now, lookup.namespace, hit_type=lookup.hit_type, key=lookup.key,
                                      similarity=lookup.similarity)
        self.last_event.update(tokens_saved=meta.total_tokens, cost_saved=meta.saved_cost)
        return response

    def on_miss(self, prompt: str, llm_response: str, **kwargs: Any) -> None:
        now = self._now(kwargs)
        namespace = self._namespace(kwargs)
        normalized = self._normalize(prompt)

        if not normalized or not str(llm_response).strip():
            self.last_event = self._event("miss", now, namespace, reason="not_cacheable")
            return

        embedding = self._last_lookup.embedding if self._last_lookup and self._last_lookup.normalized == normalized else None
        embedding = embedding or self._embed(normalized)
        if embedding is None:
            self.last_event = self._event("miss", now, namespace, reason="embedding_failure")
            return

        prompt_tokens = int(kwargs.get("prompt_tokens") or approx_tokens(normalized))
        response_tokens = int(kwargs.get("response_tokens") or approx_tokens(llm_response))
        total_tokens = int(kwargs.get("total_tokens") or prompt_tokens + response_tokens)
        latency = float(
            kwargs.get("llm_latency") or self.config.latency_base_ms + self.config.latency_per_token_ms * total_tokens)

        cost = saved_cost(latency, total_tokens)
        safe = safety_score(normalized, llm_response, kwargs)
        demand = self._demand(embedding, namespace, now)
        storage = 1.0 + total_tokens
        admission = demand * cost * safe / storage

        self.last_event = self._event("miss", now, namespace, reason="miss")
        self.last_event.update(
            admission_score=admission,
            demand_score=demand,
            saved_cost_score=cost,
            safety_score=safe,
            storage_size_score=storage,
        )

        if safe < self.config.theta_safe:
            self._add_ghost(embedding, namespace, now, cost, "rejected_unsafe")
            self.last_event.update(admission_decision="reject", admission_reason="unsafe")
            return

        meta = SRCMeta(
            key=self._key(normalized, namespace),
            prompt=normalized,
            namespace=namespace,
            embedding=embedding,
            created_at=now,
            last_access_at=now,
            saved_cost=cost,
            storage_size=storage,
            safety_score=safe,
            total_tokens=total_tokens,
            ttl_seconds=kwargs.get("ttl_seconds"),
        )

        if self.current_size() < self._max_size:
            self._insert(meta, llm_response)
            self.last_event.update(admission_decision="admit", admission_reason="free_space")
            return

        victim = self._victim(now)
        if victim is None:
            self._insert(meta, llm_response)
            self.last_event.update(admission_decision="admit", admission_reason="no_victim")
            return

        victim_meta, victim_value = victim
        self.last_event.update(victim_key=victim_meta.key, victim_eviction_value=victim_value)

        if admission > (1.0 + self.config.admission_margin) * victim_value:
            self._evict(victim_meta, now)
            self._insert(meta, llm_response)
            self.last_event.update(admission_decision="admit", admission_reason="beats_victim",
                                   evicted_key=victim_meta.key)
        else:
            self._add_ghost(embedding, namespace, now, cost, "rejected_low_score")
            self.last_event.update(admission_decision="reject", admission_reason="below_margin")

    def _insert(self, meta: SRCMeta, response: str) -> None:
        self._meta[meta.key] = meta
        self._requests_db.save(EmbeddedRequestRecord(key=meta.key, vector=meta.embedding))
        response_key = self._hash(f"{meta.key}:{response}")
        self._responses_db.save(ResponseRecord(key=response_key, request_key=meta.key, response=str(response)))

    def _evict(self, meta: SRCMeta, now: float) -> None:
        self._requests_db.remove(meta.key)
        self._responses_db.remove_by_request(meta.key)
        self._meta.pop(meta.key, None)
        self._add_ghost(meta.embedding, meta.namespace, now, meta.saved_cost, "evicted")

    def _victim(self, now: float) -> Optional[tuple[SRCMeta, float]]:
        if not self._meta:
            return None
        sample = self._rng.sample(list(self._meta.values()), min(self.config.sample_size, len(self._meta)))
        return min(((m, eviction_value(m, now, self.config)) for m in sample), key=lambda pair: pair[1])

    def _neighbors(self, embedding: list[float], namespace: str) -> list[tuple[SRCMeta, float]]:
        faiss = getattr(self._requests_db, "_faiss_client", None)
        if faiss is None or not self._meta:
            return []

        raw = faiss.fetch_nearest_k(embedding, min(max(4 * self.config.k, self.config.k), len(self._meta)))
        out: list[tuple[SRCMeta, float]] = []
        for candidate in raw:
            meta = self._meta.get(candidate.key)
            if meta and meta.namespace == namespace:
                out.append((meta, cosine(embedding, candidate.vector or meta.embedding)))
        return sorted(out, key=lambda pair: pair[1], reverse=True)[: self.config.k]

    def _demand(self, embedding: list[float], namespace: str, now: float) -> float:
        score = 1.0
        for meta, sim in self._neighbors(embedding, namespace):
            if sim >= self.config.theta_near:
                score += math.exp(-self.config.demand_decay * max(0.0, now - meta.last_access_at))

        ghosts = [g for g in self._ghosts if g.namespace == namespace]
        if len(ghosts) > self.config.ghost_sample_size:
            ghosts = self._rng.sample(ghosts, self.config.ghost_sample_size)
        for ghost in ghosts:
            if cosine(embedding, ghost.embedding) >= self.config.theta_near:
                score += self.config.ghost_weight * math.exp(-self.config.ghost_decay * max(0.0, now - ghost.timestamp))
        return score

    def _safe_reuse(self, prompt: str, namespace: str, meta: SRCMeta, sim: float, now: float) -> bool:
        return (
                meta.namespace == namespace
                and sim >= self.config.theta_hit
                and meta.safety_score >= self.config.theta_safe
                and not has_private_marker(prompt)
                and not has_time_marker(prompt)
                and self._usable(meta, now)
        )

    def _usable(self, meta: SRCMeta, now: float) -> bool:
        if meta.invalid or meta.safety_score < self.config.theta_safe:
            return False
        return meta.ttl_seconds is None or now <= meta.created_at + float(meta.ttl_seconds)

    def _add_ghost(self, embedding: list[float], namespace: str, now: float, cost: float, reason: str) -> None:
        self._ghosts.append(SRCGhost(embedding, namespace, now, cost, reason))
        limit = self.config.ghost_max_size or 0
        if 0 < limit < len(self._ghosts):
            del self._ghosts[: len(self._ghosts) - limit]

    def _embed(self, prompt: str) -> Optional[list[float]]:
        try:
            return [float(x) for x in self._embedder(prompt[: self.config.max_prompt_chars_for_embedding])]
        except Exception:
            return None

    @staticmethod
    def _normalize(prompt: str) -> str:
        return " ".join(prompt.strip().split())

    def _namespace(self, kwargs: dict[str, Any]) -> str:
        if "namespace" in kwargs:
            return self._stable(kwargs["namespace"])
        fields = {
            "model": kwargs.get("model"),
            "system_prompt": kwargs.get("system_prompt"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "tool_config": kwargs.get("tool_config"),
            "retrieval_config": kwargs.get("retrieval_config"),
            "safety_policy": kwargs.get("safety_policy"),
            "embedding_model_version": kwargs.get("embedding_model_version"),
            "user_or_tenant": kwargs.get("user_or_tenant"),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        return "default" if not fields else self._stable(fields)

    @staticmethod
    def _stable(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)

    def _key(self, prompt: str, namespace: str) -> str:
        return self._hash(f"src:{namespace}:{prompt}")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _now(kwargs: dict[str, Any]) -> float:
        try:
            return float(kwargs.get("timestamp") or time.time())
        except (TypeError, ValueError):
            return time.time()

    @staticmethod
    def _event(status: str, now: float, namespace: str, **extra: Any) -> dict[str, Any]:
        event = {
            "timestamp": now,
            "namespace": namespace,
            "hit_or_miss": status,
            "tokens_saved": 0,
            "cost_saved": 0.0,
        }
        event.update(extra)
        return event
