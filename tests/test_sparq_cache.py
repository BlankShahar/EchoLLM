import math

import numpy as np

from cache.sparq import SPARQSimilarityCache


class MappingEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return self.mapping[text]


def _miss(cache: SPARQSimilarityCache, prompt: str, response: str) -> None:
    lookup = cache.lookup(prompt)
    assert not lookup.hit
    cache.on_miss(prompt, response, lookup_context=lookup.context)


def _unit_vector(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def test_semantic_credit_prefers_the_closer_representative() -> None:
    embedder = MappingEmbedder(
        {
            "close": _unit_vector(20),
            "borderline": _unit_vector(-45),
            "query": _unit_vector(0),
        }
    )
    cache = SPARQSimilarityCache(
        max_size=2,
        hit_distance_threshold=0.3,
        prompt_embedder=embedder,
        window_fraction=0.0,
        credit_power=1.0,
        aging_interval_requests=100,
    )
    _miss(cache, "close", "answer-close")
    _miss(cache, "borderline", "answer-borderline")

    before = {resident.prompt: resident.score for resident in cache.resident_snapshot()}
    lookup = cache.lookup("query")
    after = {resident.prompt: resident.score for resident in cache.resident_snapshot()}

    assert lookup.hit
    assert lookup.response == "answer-close"
    close_credit = after["close"] - before["close"]
    borderline_credit = after["borderline"] - before["borderline"]
    assert close_credit > borderline_credit > 0.0
    assert close_credit + borderline_credit < 1.0


def test_overlapping_representatives_split_one_semantic_demand_mass() -> None:
    embedder = MappingEmbedder(
        {
            "left": _unit_vector(40),
            "right": _unit_vector(-40),
            "query": _unit_vector(0),
        }
    )
    cache = SPARQSimilarityCache(
        max_size=2,
        hit_distance_threshold=0.25,
        prompt_embedder=embedder,
        window_fraction=0.0,
        credit_power=1.0,
        aging_interval_requests=100,
    )
    _miss(cache, "left", "answer-left")
    _miss(cache, "right", "answer-right")

    before = np.asarray([resident.score for resident in cache.resident_snapshot()])
    assert cache.lookup("query").hit
    after = np.asarray([resident.score for resident in cache.resident_snapshot()])
    credits = after - before

    np.testing.assert_allclose(credits[0], credits[1], rtol=0.0, atol=1e-6)
    assert 0.0 < float(credits.sum()) < 1.0


def test_probation_rejects_one_hit_noise_and_preserves_popular_main_entries() -> None:
    prompts = ["a", "b", "c", "noise"]
    embedder = MappingEmbedder(
        {
            prompt: [
                1.0 if index == dimension else 0.0
                for dimension in range(len(prompts))
            ]
            for index, prompt in enumerate(prompts)
        }
    )
    cache = SPARQSimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.34,
        aging_interval_requests=100,
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")
    _miss(cache, "c", "answer-c")
    for _ in range(3):
        assert cache.lookup("a").hit
    for _ in range(2):
        assert cache.lookup("b").hit

    _miss(cache, "noise", "answer-noise")

    assert cache.last_decision is not None
    assert cache.last_decision.reason == "candidate_rejected"
    assert cache.last_decision.candidate_key is not None
    assert cache.lookup("a").hit
    assert cache.lookup("b").hit
    assert cache.lookup("noise").hit
    assert not cache.lookup("c").hit


def test_reused_probation_entry_is_promoted_over_a_weak_main_entry() -> None:
    prompts = ["a", "b", "candidate", "incoming"]
    embedder = MappingEmbedder(
        {
            prompt: [
                1.0 if index == dimension else 0.0
                for dimension in range(len(prompts))
            ]
            for index, prompt in enumerate(prompts)
        }
    )
    cache = SPARQSimilarityCache(
        max_size=3,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.34,
        aging_interval_requests=100,
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")
    _miss(cache, "candidate", "answer-candidate")
    assert cache.lookup("candidate").hit
    assert cache.lookup("candidate").hit

    _miss(cache, "incoming", "answer-incoming")

    assert cache.last_decision is not None
    assert cache.last_decision.reason == "candidate_promoted"
    assert cache.last_decision.promoted
    candidate = cache.lookup("candidate")
    assert candidate.hit
    assert candidate.metadata["segment"] == "main"
    assert cache.lookup("incoming").hit


def test_periodic_aging_reduces_stale_scores() -> None:
    embedder = MappingEmbedder({"a": [1.0, 0.0]})
    cache = SPARQSimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
        window_fraction=0.0,
        aging_interval_requests=3,
        aging_factor=0.5,
    )
    _miss(cache, "a", "answer-a")
    assert cache.lookup("a").hit
    score_before_aging = cache.resident_snapshot()[0].score

    assert cache.lookup("a").hit
    score_after_aging_and_credit = cache.resident_snapshot()[0].score

    assert score_before_aging == 2.0
    assert score_after_aging_and_credit == 2.0
    assert cache.stats().aging_events == 1


def test_lookup_embeds_each_framework_request_once() -> None:
    embedder = MappingEmbedder({"a": [1.0, 0.0]})
    cache = SPARQSimilarityCache(
        max_size=2,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
    )

    lookup = cache.lookup("a")
    cache.on_miss("a", "answer-a", lookup_context=lookup.context)
    assert cache.lookup("a").hit

    assert embedder.calls == 2


def test_single_slot_cache_is_a_probation_lru() -> None:
    embedder = MappingEmbedder(
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        }
    )
    cache = SPARQSimilarityCache(
        max_size=1,
        hit_distance_threshold=0.05,
        prompt_embedder=embedder,
    )
    _miss(cache, "a", "answer-a")
    _miss(cache, "b", "answer-b")

    assert cache.current_size == 1
    assert cache.lookup("b").hit
    assert not cache.lookup("a").hit
    assert cache.last_decision is not None
    assert cache.last_decision.reason == "window_replace"
