import numpy as np

from cache.sage.scorer import SAGEScorer


def test_vectorized_scores_match_brute_force() -> None:
    rng = np.random.default_rng(123)
    scorer = SAGEScorer()
    for request_count in (1, 7, 31):
        for resident_count in (1, 3, 8):
            coverage = rng.random((request_count, resident_count)) < 0.35
            coverage_counts = coverage.sum(axis=1).astype(np.int32)
            unique_owners = np.bitwise_xor.reduce(
                np.where(coverage, np.arange(resident_count), 0),
                axis=1,
            )
            candidate_covers = rng.random(request_count) < 0.4
            weights = rng.random(request_count) + 0.1
            active = np.ones(resident_count, dtype=np.bool_)
            _, _, deltas = scorer.score_all_victims(
                coverage_counts=coverage_counts,
                unique_owners=unique_owners,
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


def test_soft_top_two_scores_match_brute_force() -> None:
    rng = np.random.default_rng(321)
    scorer = SAGEScorer()
    for request_count in (1, 7, 31):
        for resident_count in (1, 3, 8):
            utilities = rng.random((request_count, resident_count))
            utilities[utilities < 0.55] = 0.0
            if resident_count == 1:
                top_values = np.column_stack(
                    [utilities[:, 0], np.zeros(request_count)]
                )
                top_slots = np.column_stack(
                    [np.zeros(request_count, dtype=np.int64), -np.ones(request_count, dtype=np.int64)]
                )
            else:
                slots = np.argpartition(utilities, -2, axis=1)[:, -2:]
                values = np.take_along_axis(utilities, slots, axis=1)
                order = np.argsort(values, axis=1)[:, ::-1]
                top_values = np.take_along_axis(values, order, axis=1)
                top_slots = np.take_along_axis(slots, order, axis=1)
            candidate = rng.random(request_count)
            candidate[candidate < 0.55] = 0.0
            weights = rng.random(request_count) + 0.1
            active = np.ones(resident_count, dtype=np.bool_)

            _, _, deltas = scorer.score_from_top_two(
                top_values=top_values,
                top_slots=top_slots,
                candidate_utilities=candidate,
                weights=weights,
                resident_active=active,
            )
            expected = np.asarray(
                [
                    scorer.brute_force_utility_delta(
                        utilities=utilities,
                        candidate_utilities=candidate,
                        weights=weights,
                        victim_slot=slot,
                    )
                    for slot in range(resident_count)
                ]
            )
            np.testing.assert_allclose(deltas, expected, atol=1e-12)
