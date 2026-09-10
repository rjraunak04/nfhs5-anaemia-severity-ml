"""Leakage-safe development, calibration, and locked-test partitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from anaemia_ml.evaluation.grouped_validation import GroupLeakageError


class PartitionError(ValueError):
    """Raised when safe grouped partitions cannot be created."""


@dataclass(frozen=True)
class GroupedPartitions:
    """Row indices for development, calibration, and locked-test data."""

    development: np.ndarray
    calibration: np.ndarray
    locked_test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return partition indices keyed by their scientific role."""
        return {
            "development": self.development,
            "calibration": self.calibration,
            "locked_test": self.locked_test,
        }


def _one_dimensional(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim != 1:
        raise PartitionError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise PartitionError(f"{name} must not be empty.")

    return array


def _fold_count(fraction: float, *, name: str) -> int:
    reciprocal = 1.0 / fraction
    folds = round(reciprocal)

    if folds < 2 or not math.isclose(
        reciprocal,
        folds,
        abs_tol=1e-9,
    ):
        raise PartitionError(
            f"{name} must be the reciprocal of an integer >= 2 "
            "for StratifiedGroupKFold."
        )

    return folds


def _distribution(
    y_encoded: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    counts = np.bincount(
        y_encoded[indices],
        minlength=n_classes,
    )
    return counts / counts.sum()


def _candidate_score(
    partitions: GroupedPartitions,
    y_encoded: np.ndarray,
    target_fractions: np.ndarray,
    global_distribution: np.ndarray,
    n_classes: int,
) -> float:
    indices = tuple(partitions.as_dict().values())

    actual_fractions = np.asarray(
        [partition.size / y_encoded.size for partition in indices]
    )
    size_error = np.mean(
        np.abs(actual_fractions - target_fractions)
        / target_fractions
    )

    distribution_error = np.mean(
        [
            np.abs(
                _distribution(
                    y_encoded,
                    partition,
                    n_classes,
                )
                - global_distribution
            ).sum()
            for partition in indices
        ]
    )

    return float(size_error + distribution_error)


def assert_group_disjoint_partitions(
    partitions: GroupedPartitions,
    groups: Any,
) -> None:
    """Raise when a PSU group appears in more than one partition."""
    groups_array = _one_dimensional(groups, name="groups")

    group_sets = {
        name: set(groups_array[indices].tolist())
        for name, indices in partitions.as_dict().items()
    }

    for (
        left_name,
        left_groups,
    ), (
        right_name,
        right_groups,
    ) in combinations(group_sets.items(), 2):
        overlap = left_groups & right_groups

        if overlap:
            raise GroupLeakageError(
                f"Group leakage between {left_name} and "
                f"{right_name}: {len(overlap)} overlapping groups."
            )


def make_grouped_partitions(
    y: Any,
    groups: Any,
    *,
    development_fraction: float = 0.70,
    calibration_fraction: float = 0.10,
    locked_test_fraction: float = 0.20,
    random_state: int = 42,
) -> GroupedPartitions:
    """Create deterministic, class-aware, group-disjoint partitions."""
    y_array = _one_dimensional(y, name="y")
    groups_array = _one_dimensional(groups, name="groups")

    if y_array.size != groups_array.size:
        raise PartitionError(
            "y and groups must contain the same number of rows."
        )

    if isinstance(random_state, bool) or not isinstance(
        random_state,
        int,
    ):
        raise PartitionError("random_state must be an integer.")

    fractions = np.asarray(
        [
            development_fraction,
            calibration_fraction,
            locked_test_fraction,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0):
        raise PartitionError(
            "All partition fractions must be positive and finite."
        )

    if not math.isclose(
        float(fractions.sum()),
        1.0,
        abs_tol=1e-9,
    ):
        raise PartitionError(
            "Partition fractions must sum to 1.0."
        )

    locked_folds = _fold_count(
        locked_test_fraction,
        name="locked_test_fraction",
    )

    remaining_fraction = (
        development_fraction + calibration_fraction
    )
    calibration_relative = (
        calibration_fraction / remaining_fraction
    )
    calibration_folds = _fold_count(
        calibration_relative,
        name="calibration fraction relative to non-test data",
    )

    classes, y_encoded = np.unique(
        y_array,
        return_inverse=True,
    )

    if classes.size < 2:
        raise PartitionError(
            "y must contain at least two classes."
        )

    if np.unique(groups_array).size < locked_folds + 1:
        raise PartitionError(
            "Too few groups for the requested partition fractions."
        )

    all_rows = np.arange(y_array.size)
    global_distribution = _distribution(
        y_encoded,
        all_rows,
        classes.size,
    )

    outer_splitter = StratifiedGroupKFold(
        n_splits=locked_folds,
        shuffle=True,
        random_state=random_state,
    )

    dummy_features = np.zeros(
        (y_array.size, 1),
        dtype=np.uint8,
    )

    best: GroupedPartitions | None = None
    best_score = math.inf

    try:
        outer_candidates = outer_splitter.split(
            dummy_features,
            y_encoded,
            groups_array,
        )

        for outer_number, (
            remaining,
            locked_test,
        ) in enumerate(outer_candidates):
            inner_splitter = StratifiedGroupKFold(
                n_splits=calibration_folds,
                shuffle=True,
                random_state=random_state + outer_number + 1,
            )

            inner_features = np.zeros(
                (remaining.size, 1),
                dtype=np.uint8,
            )

            inner_candidates = inner_splitter.split(
                inner_features,
                y_encoded[remaining],
                groups_array[remaining],
            )

            for (
                development_relative,
                calibration_relative_indices,
            ) in inner_candidates:
                candidate = GroupedPartitions(
                    development=np.sort(
                        remaining[development_relative]
                    ),
                    calibration=np.sort(
                        remaining[calibration_relative_indices]
                    ),
                    locked_test=np.sort(locked_test),
                )

                if any(
                    np.unique(y_encoded[indices]).size
                    != classes.size
                    for indices in candidate.as_dict().values()
                ):
                    continue

                score = _candidate_score(
                    candidate,
                    y_encoded,
                    fractions,
                    global_distribution,
                    classes.size,
                )

                if score < best_score:
                    best = candidate
                    best_score = score

    except ValueError as error:
        raise PartitionError(
            "Unable to construct stratified grouped partitions."
        ) from error

    if best is None:
        raise PartitionError(
            "No valid partition preserved every outcome class. "
            "Ensure each class spans enough PSU groups."
        )

    combined = np.concatenate(
        tuple(best.as_dict().values())
    )

    if combined.size != y_array.size or not np.array_equal(
        np.sort(combined),
        all_rows,
    ):
        raise PartitionError(
            "Generated partitions do not cover each row exactly once."
        )

    assert_group_disjoint_partitions(
        best,
        groups_array,
    )
    return best