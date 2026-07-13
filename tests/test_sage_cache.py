from pathlib import Path

import numpy as np

from cache.sage import SAGESimilarityCache
from cache.similarity_cache import RankingDistanceMethod


class MappingEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return self.mapping[text]


def _miss(cache: SAGESimilarityCache, prompt: str, response: str) -> None:
    lookup = cache.lookup(prompt)
    assert not lookup.hit
    cache.on_miss(prompt, response, lookup_context=lookup.context)


def test_sage_rejects_one_hit_noise_after_popular_regions_are_observed() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "noise": [0.0, 0.0, 1.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        ghost_capacity=32,
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")

    for _ in range(4):
        assert cache.lookup("a").hit
        assert cache.lookup("b").hit

    _miss(cache, "noise", "answer-noise")
    assert cache.current_size == 2
    assert cache.last_decision is not None
    assert not cache.last_decision.admitted
    assert cache.last_decision.reason == "non_positive_gain"
    assert cache.lookup("a").hit
    assert cache.lookup("b").hit


def test_sage_admits_repeated_uncovered_demand() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        ghost_capacity=64,
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")
    assert cache.lookup("a").hit
    assert cache.lookup("b").hit

    first = cache.lookup("c")
    assert not first.hit
    cache.on_miss("c", "answer-c", lookup_context=first.context)
    assert cache.last_decision is not None

    # Rejected misses remain in the ghost window and accumulate evidence.
    for _ in range(4):
        lookup = cache.lookup("c")
        if lookup.hit:
            break
        cache.on_miss("c", "answer-c", lookup_context=lookup.context)
    assert cache.lookup("c").hit


def test_sqlite_persistence_round_trip(tmp_path: Path) -> None:
    mapping = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path = tmp_path / "sage.sqlite3"
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(mapping),
        storage_path=path,
        storage_namespace="test",
    )
    _miss(cache, "a", "answer-a")
    cache.close()

    restored = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(mapping),
        storage_path=path,
        storage_namespace="test",
    )
    lookup = restored.lookup("a")
    assert lookup.hit
    assert lookup.response == "answer-a"
    restored.close()


def test_euclidean_distance_is_supported() -> None:
    embedder = MappingEmbedder({"a": [0.0, 0.0], "near": [0.1, 0.0]})
    cache = SAGESimilarityCache(
        max_size=1,
        hit_distance_threshold=0.2,
        prompt_embedder=embedder,
        ranking_distance_method=RankingDistanceMethod.EUCLIDEAN,
    )
    _miss(cache, "a", "answer")
    assert cache.lookup("near").hit


def test_victim_selection_uses_the_exact_best_delta() -> None:
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=lambda _: [1.0, 0.0],
    )
    cache._active[:] = True
    cache._last_access_steps[:] = [0, 10]

    assert cache._choose_victim(np.asarray([1.0, 1.000005])) == 1
