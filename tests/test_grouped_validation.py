"""Tests for leakage-safe nested PSU-grouped validation."""

from __future__ import annotations

import numpy as np
import pytest

from anaemia_ml.evaluation import (
    GroupLeakageError,
    assert_disjoint_groups,
    nested_group_splits,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Create four balanced classes across 24 synthetic PSUs."""
    groups = np.repeat(np.arange(24), 4)
    target = np.tile(np.arange(4), 24)
    return target, groups


def test_nested_folds_are_group_disjoint(
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    target, groups = synthetic_data

    folds = nested_group_splits(
        target,
        groups,
        outer_splits=5,
        inner_splits=3,
        random_state=42,
    )

    assert len(folds) == 5
    validation_counts = np.zeros(target.size, dtype=int)

    for fold in folds:
        outer_train = fold.outer.train_indices
        outer_validation = fold.outer.validation_indices

        train_groups = set(groups[outer_train].tolist())
        validation_groups = set(groups[outer_validation].tolist())

        assert train_groups.isdisjoint(validation_groups)
        assert len(fold.inner) == 3

        validation_counts[outer_validation] += 1
        outer_train_rows = set(outer_train.tolist())

        for inner_fold in fold.inner:
            inner_train = inner_fold.train_indices
            inner_validation = inner_fold.validation_indices

            assert set(inner_train.tolist()).issubset(outer_train_rows)
            assert set(inner_validation.tolist()).issubset(outer_train_rows)

            inner_train_groups = set(groups[inner_train].tolist())
            inner_validation_groups = set(groups[inner_validation].tolist())

            assert inner_train_groups.isdisjoint(inner_validation_groups)

    assert np.all(validation_counts == 1)


def test_nested_folds_are_reproducible(
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    target, groups = synthetic_data

    first = nested_group_splits(target, groups, random_state=42)
    second = nested_group_splits(target, groups, random_state=42)

    for first_fold, second_fold in zip(first, second, strict=True):
        assert np.array_equal(
            first_fold.outer.train_indices,
            second_fold.outer.train_indices,
        )
        assert np.array_equal(
            first_fold.outer.validation_indices,
            second_fold.outer.validation_indices,
        )


def test_group_overlap_is_rejected() -> None:
    with pytest.raises(GroupLeakageError, match="1 overlapping groups"):
        assert_disjoint_groups(
            train_groups=[101, 102, 103],
            validation_groups=[103, 104],
            context="outer fold 1",
        )


def test_different_input_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        nested_group_splits(
            y=[0, 1, 2, 3],
            groups=[10, 11, 12],
        )


def test_missing_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not contain missing values"):
        nested_group_splits(
            y=[0, 1, 0, 1],
            groups=[10, 11, np.nan, 13],
            outer_splits=2,
            inner_splits=2,
        )