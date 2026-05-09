"""
models.py — Data structures for the Semantic Resonance Cache.

SRC stores only *decision metadata* alongside EchoLLM's existing request /
response databases.  Prompts, embeddings, and responses are never duplicated.
"""

from collections import deque
from typing import Deque, Literal, Self

from pydantic import BaseModel, model_validator


class SRCItemMeta(BaseModel):
    """
    Lightweight metadata record for one cached item.

    Attributes
    ----------
    normalised_cost : float
        Normalised saved-generation-cost score  Ĉ_i ∈ [0, 1).
        Computed once at admission from (latency, token_count).
    safety : float
        Safety-and-reusability score  R_i ∈ [0, 1].
        Computed once at admission using deterministic heuristics.
    """

    normalised_cost: float  # Ĉ_i  ∈ [0, 1)
    safety: float  # R_i  ∈ [0, 1]

    @model_validator(mode='after')
    def validate(self) -> Self:
        if not (0.0 <= self.normalised_cost < 1.0):
            raise ValueError(f"normalised_cost must be in [0, 1), got {self.normalised_cost}")
        if not (0.0 <= self.safety <= 1.0):
            raise ValueError(f"safety must be in [0, 1], got {self.safety}")
        return self


GhostReason = Literal["rejected", "evicted"]


class GhostEntry(BaseModel):
    """
    A ghost record for a semantic region that was rejected or evicted.

    The embedding is stored as a plain tuple so it is hashable and can be used
    with ``@lru_cache`` helpers downstream.  It is the *normalised* L2 vector
    so that cosine similarity reduces to a dot product.

    Attributes
    ----------
    embedding : tuple[float, ...]
        L2-normalised prompt embedding.
    reason : GhostReason
        Why this region was added to the ghost history.
    """

    embedding: tuple[float, ...]
    reason: GhostReason


class GhostHistory:
    """
    Bounded FIFO ghost history with O(1) insertion and O(n) similarity scan.

    Capacity is always ``2 * cache_capacity`` (G_max = 2B).
    When full the *oldest* entry is dropped on insertion.

    Parameters
    ----------
    cache_capacity : int
        Maximum number of items in the SRC cache (B).
    """

    def __init__(self, cache_capacity: int) -> None:
        self._max_size: int = 2 * cache_capacity
        self._entries: Deque[GhostEntry] = deque()

    def add(self, embedding: tuple[float, ...], reason: GhostReason) -> None:
        """Insert a new ghost entry, evicting the oldest one if at capacity."""
        if len(self._entries) >= self._max_size:
            self._entries.popleft()
        self._entries.append(GhostEntry(embedding=embedding, reason=reason))

    def clear(self) -> None:
        """Remove all ghost entries (useful for testing)."""
        self._entries.clear()

    def entries(self) -> list[GhostEntry]:
        """Return a snapshot of all ghost entries (oldest first)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"GhostHistory(size={len(self)}, max={self._max_size})"
