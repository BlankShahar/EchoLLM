import numpy as np

from cache.sage.ghost_window import GhostWindow


def test_unique_owner_is_maintained_without_scanning_resident_columns() -> None:
    window = GhostWindow(capacity=4, resident_capacity=4)
    window.add(
        np.asarray([1.0, 0.0], dtype=np.float32),
        step=1,
        coverage=np.asarray([True, False, True, False]),
    )
    window.add(
        np.asarray([0.0, 1.0], dtype=np.float32),
        step=2,
        coverage=np.asarray([False, True, False, False]),
    )

    np.testing.assert_array_equal(window.active_coverage_counts(), [2, 1])
    assert window.active_unique_owners()[1] == 1

    window.replace_resident_column(2, np.asarray([False, False]))

    np.testing.assert_array_equal(window.active_coverage_counts(), [1, 1])
    np.testing.assert_array_equal(window.active_unique_owners(), [0, 1])


def test_current_observation_can_be_discounted_for_admission_only() -> None:
    window = GhostWindow(capacity=4, resident_capacity=1)
    first_id = window.add(
        np.asarray([1.0], dtype=np.float32),
        step=1,
        coverage=np.asarray([False]),
    )
    current_id = window.add(
        np.asarray([1.0], dtype=np.float32),
        step=2,
        coverage=np.asarray([False]),
    )

    weights = window.weights(
        current_step=2,
        half_life_requests=None,
        current_observation_id=current_id,
        current_observation_weight=0.1,
    )

    np.testing.assert_allclose(weights, [1.0, 0.1])
    assert first_id != current_id
