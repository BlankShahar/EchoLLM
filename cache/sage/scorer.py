import numpy as np


class SAGEScorer:
    """Vectorized exact counterfactual replacement scorer."""

    @staticmethod
    def score_all_victims(
        *,
        coverage_counts: np.ndarray,
        unique_owners: np.ndarray,
        candidate_covers: np.ndarray,
        weights: np.ndarray,
        resident_active: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if coverage_counts.ndim != 1:
            raise ValueError("coverage_counts must be one-dimensional")
        request_count = coverage_counts.shape[0]
        resident_capacity = resident_active.shape[0]
        expected_shape = (request_count,)
        if unique_owners.shape != expected_shape:
            raise ValueError("unique_owners shape mismatch")
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
        unrecovered_owners = unique_owners[unrecovered]
        if unrecovered_owners.size and (
            int(unrecovered_owners.min()) < 0
            or int(unrecovered_owners.max()) >= resident_capacity
        ):
            raise ValueError("unique owner is outside the resident capacity")
        victim_losses = np.bincount(
            unrecovered_owners,
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

    @staticmethod
    def score_from_top_two(
        *,
        top_values: np.ndarray,
        top_slots: np.ndarray,
        candidate_utilities: np.ndarray,
        weights: np.ndarray,
        resident_active: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Score max-utility replacement exactly from each row's best two residents."""
        request_count = candidate_utilities.shape[0]
        if top_values.shape != (request_count, 2):
            raise ValueError("top_values shape mismatch")
        if top_slots.shape != (request_count, 2):
            raise ValueError("top_slots shape mismatch")
        if weights.shape != (request_count,):
            raise ValueError("weights shape mismatch")

        best = top_values[:, 0].astype(np.float64, copy=False)
        second = top_values[:, 1].astype(np.float64, copy=False)
        candidate = candidate_utilities.astype(np.float64, copy=False)
        candidate_best = np.maximum(candidate, best)
        base_improvements = weights * (candidate_best - best)
        new_gain = float(base_improvements.sum())

        corrections = weights * (np.maximum(candidate, second) - candidate_best)
        owners = top_slots[:, 0]
        valid = (best > second + 1e-12) & (owners >= 0)
        if bool(valid.any()):
            if int(owners[valid].max()) >= resident_active.shape[0]:
                raise ValueError("top owner is outside resident capacity")
            victim_corrections = np.bincount(
                owners[valid],
                weights=corrections[valid],
                minlength=resident_active.shape[0],
            ).astype(np.float64, copy=False)
        else:
            victim_corrections = np.zeros(resident_active.shape[0], dtype=np.float64)

        deltas = new_gain + victim_corrections
        victim_losses = -victim_corrections
        deltas[~resident_active] = -np.inf
        victim_losses[~resident_active] = np.inf
        return new_gain, victim_losses, deltas

    @staticmethod
    def brute_force_utility_delta(
        *,
        utilities: np.ndarray,
        candidate_utilities: np.ndarray,
        weights: np.ndarray,
        victim_slot: int,
    ) -> float:
        before = utilities.max(axis=1)
        after_utilities = utilities.copy()
        after_utilities[:, victim_slot] = candidate_utilities
        after = after_utilities.max(axis=1)
        return float(np.sum(weights * (after - before)))
