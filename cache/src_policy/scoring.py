"""
scoring.py — Cost, demand, and unified SRC score computation.

All formulas are direct translations from the project spec.

Notation (from spec)
--------------------
  L_i   — measured LLM latency at miss time (milliseconds)
  T_i   — token count of (prompt + response)
  C_i   — raw saved cost = L_i * log(1 + T_i)
  Ĉ_i   — normalised cost  = C_i / (1 + C_i)   ∈ [0, 1)
  D(p)  — raw semantic demand  ∈ [1, 1 + 2k]
  D̂(p)  — normalised demand   = D(p) / (1 + 2k)  ∈ (0, 1]
  R_i   — safety score        ∈ {0.0, 0.3, 0.7, 1.0}
  S_i   — unified SRC score   = D̂_i * Ĉ_i * R_i
"""

import math
from typing import Sequence

import numpy as np


def estimate_tokens(prompt: str, response: str) -> int:
    """
    Estimate total token count for the (prompt + response) pair.

    Uses the rough heuristic  T ≈ (|prompt| + |response|) / 4  when a real
    tokenizer is not available.  The minimum is 1 to keep log(1 + T) > 0.
    """
    return max(1, (len(prompt) + len(response)) // 4)


def compute_normalised_cost(latency_ms: float, token_count: int) -> float:
    """
    Compute the normalised saved-cost score  Ĉ ∈ [0, 1).

    Formula
    -------
        raw  = L * log(1 + T)
        Ĉ    = raw / (1 + raw)

    Parameters
    ----------
    latency_ms : float
        Measured LLM latency in **milliseconds** (as reported by EchoLLM).
    token_count : int
        Total token count for the prompt + response pair (≥ 1).

    Returns
    -------
    float
        Ĉ ∈ [0, 1).  Returns 0.0 when latency or tokens are non-positive.
    """
    if latency_ms <= 0.0 or token_count <= 0:
        return 0.0

    raw = latency_ms * math.log1p(token_count)  # L * log(1 + T)
    return raw / (1.0 + raw)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Exact cosine similarity between two vectors.

    Returns 0.0 when either vector is the zero vector.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    normalised_a = float(np.linalg.norm(arr_a))
    normalised_b = float(np.linalg.norm(arr_b))
    if normalised_a == 0.0 or normalised_b == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (normalised_a * normalised_b))


def _cosine_similarity_normalized(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Fast cosine similarity when *both* vectors are already L2-normalised.

    Reduces to a dot product, which is O(d) with no division.
    """
    return float(np.dot(np.asarray(a, dtype=np.float64),
                        np.asarray(b, dtype=np.float64)))


def normalize_embedding(vec: Sequence[float]) -> tuple[float, ...]:
    """L2-normalise *vec* and return it as an immutable tuple."""
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return tuple(arr.tolist())
    return tuple((arr / norm).tolist())


def compute_demand(
        query_embedding: Sequence[float],
        cache_embeddings: list[Sequence[float]],
        ghost_embeddings: list[Sequence[float]],
        theta_near: float,
) -> float:
    """
    Compute the raw semantic demand  D(p)  for prompt *p*.

    Formula (spec §10)
    ------------------
        D(p) = 1
             + Σ_{j ∈ TopK_cache(p)}  1[ sim(p, p_j) ≥ θ_near ]
             + Σ_{g ∈ TopK_ghost(p)}  1[ sim(p, p_g) ≥ θ_near ]

    Parameters
    ----------
    query_embedding : Sequence[float]
        Raw (not necessarily normalised) embedding of the query prompt.
    cache_embeddings : list[Sequence[float]]
        Up to *k* candidate embeddings from the cache (already retrieved
        from the FAISS index — the caller is responsible for the TopK fetch).
    ghost_embeddings : list[Sequence[float]]
        Up to *k* candidate embeddings from the ghost history (also pre-
        retrieved by the caller).
    theta_near : float
        Minimum cosine similarity for a neighbour to count as "nearby"
        (spec default 0.80).

    Returns
    -------
    float
        Raw demand D(p) ∈ [1, 1 + 2k].
    """
    demand = 1  # cold-start base

    for emb in cache_embeddings:
        if _cosine_similarity(query_embedding, emb) >= theta_near:
            demand += 1

    for emb in ghost_embeddings:
        if _cosine_similarity(query_embedding, emb) >= theta_near:
            demand += 1

    return float(demand)


def normalize_demand(raw_demand: float, k: int) -> float:
    """
    Normalise raw demand to  D̂ ∈ (0, 1].

    Formula: D̂ = D / (1 + 2k)
    """
    normaliser = 1.0 + 2.0 * k
    return raw_demand / normaliser


def src_score(normalised_demand: float, normalised_cost: float, safety: float) -> float:
    """
    Compute the unified SRC score  S = D̂ · Ĉ · R.

    All three factors are already normalised to (0, 1].
    The product is high only when the prompt is in a dense semantic region,
    the response was expensive to regenerate, *and* it is safe to reuse.

    Returns
    -------
    float
        S ∈ [0, 1).
    """
    return normalised_demand * normalised_cost * safety
