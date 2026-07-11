import numpy as np


class SAGEScorer:
    """Vectorized exact counterfactual replacement scorer."""

    @staticmethod
    def score_all_victims(
        *,
        coverage: np.ndarray,
        coverage_counts: np.ndarray,
        candidate_covers: np.ndarray,
        weights: np.ndarray,
        resident_active: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        request_count, resident_capacity = coverage.shape
        expected_shape = (request_count,)
        if coverage_counts.shape != expected_shape:
            raise ValueError("coverage_counts shape mismatch")
        if candidate_covers.shape != expected_shape:
            raise ValueError("candidate_covers shape mismatch")
        if weights.shape != expected_shape:
            raise ValueError("weights shape mismatch")
        if resident_active.shape != (resident_capacity,):
            raise ValueError("resident_active shape mismatch")

        newly_covered = (coverage_counts == 0) & candidate_covers
        new_gain = float(weights[newly_covered].sum())

        uniquely_covered = coverage_counts == 1
        unrecovered = uniquely_covered & ~candidate_covers
        unique_owners = np.argmax(coverage, axis=1)
        victim_losses = np.bincount(
            unique_owners[unrecovered],
            weights=weights[unrecovered],
            minlength=resident_capacity,
        ).astype(np.float64, copy=False)
        victim_losses[~resident_active] = np.inf
        deltas = new_gain - victim_losses
        return new_gain, victim_losses, deltas

    @staticmethod
    def brute_force_delta(
        *,
        coverage: np.ndarray,
        candidate_covers: np.ndarray,
        weights: np.ndarray,
        victim_slot: int,
    ) -> float:
        before = coverage.any(axis=1)
        after_coverage = coverage.copy()
        after_coverage[:, victim_slot] = candidate_covers
        after = after_coverage.any(axis=1)
        return float(weights[after].sum() - weights[before].sum())
