from pathlib import Path

import numpy as np

from cache.sage import SAGESimilarityCache
from cache.sage.models import PersistedResident, SAGEStorageMetadata
from cache.sage.storage import SAGEStorage
from cache.similarity_cache import RankingDistanceMethod


class MappingEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return self.mapping[text]


class FailingStorage(SAGEStorage):
    def __init__(self) -> None:
        self.residents: dict[int, PersistedResident] = {}
        self.fail = False

    def load(self) -> list[PersistedResident]:
        return list(self.residents.values())

    def apply(
        self,
        residents: list[PersistedResident],
        deleted_slots: tuple[int, ...] = (),
    ) -> None:
        if self.fail:
            raise RuntimeError("injected storage failure")
        for slot in deleted_slots:
            self.residents.pop(slot, None)
        self.residents.update({resident.slot: resident for resident in residents})

    def clear(self) -> None:
        self.residents.clear()

    def validate_or_initialize(self, metadata: SAGEStorageMetadata) -> None:
        return None


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
        window_fraction=0.0,
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
        window_fraction=0.0,
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
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(mapping),
        window_fraction=0.34,
        storage_path=path,
        storage_namespace="test",
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")
    assert cache.lookup("a").hit
    cache.close()

    restored = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(mapping),
        window_fraction=0.34,
        storage_path=path,
        storage_namespace="test",
    )
    lookup = restored.lookup("a")
    assert lookup.hit
    assert lookup.response == "answer-a"
    assert max(restored._frequencies) == 3
    assert restored.lookup("b").hit
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
        window_fraction=0.0,
    )
    cache._active[:] = True
    cache._last_access_steps[:] = [0, 10]

    assert cache._choose_victim(np.asarray([1.0, 1.000005])) == 1


def test_frequency_regularizer_protects_directly_reused_residents() -> None:
    cache = SAGESimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=lambda _: [1.0, 0.0],
        window_fraction=0.0,
        frequency_weight=1.0,
    )
    cache._active[:] = True
    cache._frequencies[:] = [10, 1]
    cache._step = 10

    adjusted = cache._frequency_adjusted_deltas(
        np.zeros(2, dtype=np.float64),
        candidate_frequency=1,
    )

    assert cache._choose_victim(adjusted) == 1


def test_wsage_probation_window_catches_immediate_reuse() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=4,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.25,
        ghost_capacity=16,
        long_history_capacity=16,
    )

    _miss(cache, "a", "answer-a")

    lookup = cache.lookup("a")
    assert lookup.hit
    assert lookup.response == "answer-a"
    assert lookup.metadata["segment"] == "window"
    assert cache.stats().window_hits == 1


def test_wsage_moves_window_lru_to_free_main_slot() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.34,
        ghost_capacity=16,
        long_history_capacity=16,
    )

    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")

    a = cache.lookup("a")
    b = cache.lookup("b")
    assert a.hit and a.metadata["segment"] == "main"
    assert b.hit and b.metadata["segment"] == "window"
    assert cache.current_size == 2
    assert cache.stats().promotions == 1


def test_wsage_discounts_the_probation_candidates_original_miss() -> None:
    prompts = ["a", "b", "c", "d"]
    cache = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(
            {
                prompt: [1.0 if index == dimension else 0.0 for dimension in range(4)]
                for index, prompt in enumerate(prompts)
            }
        ),
        window_fraction=0.34,
        ghost_capacity=16,
        long_history_capacity=16,
        current_request_weight=0.1,
    )
    for prompt in prompts[:3]:
        _miss(cache, prompt, f"answer-{prompt}")

    captured_observation_ids: list[int | None] = []
    original_score_candidate = cache._score_candidate

    def capture_score(
        vector: np.ndarray,
        *,
        current_observation_id: int | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        captured_observation_ids.append(current_observation_id)
        return original_score_candidate(
            vector, current_observation_id=current_observation_id
        )

    cache._score_candidate = capture_score  # type: ignore[method-assign]
    _miss(cache, "d", "answer-d")

    assert captured_observation_ids == [2]


def test_fractional_responsibility_breaks_delta_tie_before_lru() -> None:
    cache = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.5,
        prompt_embedder=lambda _: [1.0, 0.0],
        window_fraction=0.0,
        soft_coverage=False,
        ghost_capacity=8,
        long_history_capacity=8,
    )
    cache._ensure_dimension(2)
    cache._active[:] = True
    cache._last_access_steps[:] = [0, 10, 20]
    cache._recent.add(
        np.asarray([1.0, 0.0], dtype=np.float32),
        step=1,
        coverage=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    )
    cache._recent.add(
        np.asarray([0.0, 1.0], dtype=np.float32),
        step=2,
        coverage=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    # All objective deltas tie, but slot 2 has zero fractional responsibility.
    assert cache._choose_victim(np.zeros(3, dtype=np.float64)) == 2


def test_soft_coverage_is_enabled_by_default_and_configurable() -> None:
    default = SAGESimilarityCache(
        max_size=4,
        hit_distance_threshold=0.5,
        prompt_embedder=lambda _: [1.0, 0.0],
    )
    binary = SAGESimilarityCache(
        max_size=4,
        hit_distance_threshold=0.5,
        prompt_embedder=lambda _: [1.0, 0.0],
        soft_coverage=False,
    )

    assert default.config.soft_coverage is True
    np.testing.assert_allclose(default._distance_utilities(np.asarray([0.0, 0.25, 0.5])), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(binary._distance_utilities(np.asarray([0.0, 0.25, 0.5])), [1.0, 1.0, 1.0])


def test_recent_evictions_are_downsampled_into_long_evidence() -> None:
    mapping = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
    }
    cache = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=MappingEmbedder(mapping),
        ghost_capacity=1,
        long_history_capacity=4,
        long_sample_stride=1,
    )

    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")
    _miss(cache, "c", "answer-c")

    assert cache._recent.size == 1
    assert cache._long.size == 2


def test_window_rotation_rolls_back_all_slots_when_storage_fails() -> None:
    storage = FailingStorage()
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.34,
        ghost_capacity=8,
        long_history_capacity=8,
        storage=storage,
    )
    _miss(cache, "a", "answer-a")
    before = cache.resident_snapshot()
    storage.fail = True

    lookup = cache.lookup("b")
    try:
        cache.on_miss("b", "answer-b", lookup_context=lookup.context)
    except RuntimeError as error:
        assert str(error) == "injected storage failure"
    else:
        raise AssertionError("storage failure was not propagated")

    assert cache.resident_snapshot() == before
    assert cache.current_size == 1
    assert cache.lookup("a").hit


def test_single_slot_cache_behaves_as_pure_probation_lru() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        }
    )
    cache = SAGESimilarityCache(
        max_size=1,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        ghost_capacity=4,
        long_history_capacity=4,
    )

    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")

    assert cache.current_size == 1
    assert cache.lookup("b").hit
    assert not cache.lookup("a").hit
    assert cache.last_decision is not None
    assert cache.last_decision.incoming_admitted is True
    assert cache.last_decision.promoted is False
