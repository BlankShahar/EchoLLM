import numpy as np

from cache.sage.scorer import SAGEScorer


def test_vectorized_scores_match_brute_force() -> None:
    rng = np.random.default_rng(123)
    scorer = SAGEScorer()
    for request_count in (1, 7, 31):
        for resident_count in (1, 3, 8):
            coverage = rng.random((request_count, resident_count)) < 0.35
            coverage_counts = coverage.sum(axis=1).astype(np.int32)
            candidate_covers = rng.random(request_count) < 0.4
            weights = rng.random(request_count) + 0.1
            active = np.ones(resident_count, dtype=np.bool_)
            _, _, deltas = scorer.score_all_victims(
                coverage=coverage,
                coverage_counts=coverage_counts,
                candidate_covers=candidate_covers,
                weights=weights,
                resident_active=active,
            )
            expected = np.asarray(
                [
                    scorer.brute_force_delta(
                        coverage=coverage,
                        candidate_covers=candidate_covers,
                        weights=weights,
                        victim_slot=slot,
                    )
                    for slot in range(resident_count)
                ]
            )
            np.testing.assert_allclose(deltas, expected, atol=1e-12)
