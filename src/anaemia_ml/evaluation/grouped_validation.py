"""Create leakage-safe nested cross-validation folds grouped by PSU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import StratifiedGroupKFold


class GroupLeakageError(ValueError):
    """Raised when the same PSU/group occurs on both sides of a split."""


@dataclass(frozen=True)
class GroupedFold:
    """Global row indices for one train-validation fold."""

    train_indices: NDArray[np.intp]
    validation_indices: NDArray[np.intp]


@dataclass(frozen=True)
class NestedGroupedFold:
    """One outer fold and its inner folds."""

    number: int
    outer: GroupedFold
    inner: tuple[GroupedFold, ...]


def _as_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert input to a validated one-dimensional array."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if bool(pd.isna(array).any()):
        raise ValueError(f"{name} must not contain missing values.")

    return array


def _validate_split_count(value: int, *, name: str) -> None:
    if value < 2:
        raise ValueError(f"{name} must be at least 2.")


def _require_class_group_support(
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    context: str,
) -> None:
    """Ensure every outcome class occurs in enough independent groups."""
    for label in pd.unique(target):
        group_count = int(pd.unique(groups[target == label]).size)

        if group_count < n_splits:
            raise ValueError(
                f"{context}: class {label!r} occurs in {group_count} groups; "
                f"at least {n_splits} are required."
            )


def assert_disjoint_groups(
    train_groups: ArrayLike,
    validation_groups: ArrayLike,
    *,
    context: str = "split",
) -> None:
    """Raise an error if a PSU occurs in both partitions."""
    train = _as_vector(train_groups, name="train_groups")
    validation = _as_vector(validation_groups, name="validation_groups")

    overlap = set(pd.unique(train)).intersection(pd.unique(validation))

    if overlap:
        examples = sorted(str(value) for value in overlap)[:5]
        raise GroupLeakageError(
            f"{context} contains {len(overlap)} overlapping groups; "
            f"examples: {examples}."
               )


def nested_group_splits(
    y: ArrayLike,
    groups: ArrayLike,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> tuple[NestedGroupedFold, ...]:
    """Create reproducible stratified nested-CV folds without PSU leakage.

    Inner and outer indices refer to the original input rows. This function
    should receive development data only. Calibration and locked-test rows
    must remain outside this process.
    """
    target = _as_vector(y, name="y")
    group_values = _as_vector(groups, name="groups")

    if target.size != group_values.size:
        raise ValueError("y and groups must have the same length.")

    _validate_split_count(outer_splits, name="outer_splits")
    _validate_split_count(inner_splits, name="inner_splits")

    if pd.unique(group_values).size < outer_splits:
        raise ValueError("Not enough unique groups for outer_splits.")

    _require_class_group_support(
        target,
        group_values,
        n_splits=outer_splits,
        context="outer CV",
    )

    placeholder = np.zeros(target.size, dtype=np.uint8)

    outer_splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    nested_folds: list[NestedGroupedFold] = []

    for outer_number, (outer_train, outer_validation) in enumerate(
        outer_splitter.split(placeholder, target, group_values),
        start=1,
    ):
        assert_disjoint_groups(
            group_values[outer_train],
            group_values[outer_validation],
            context=f"outer fold {outer_number}",
        )

        outer_target = target[outer_train]
        outer_groups = group_values[outer_train]

        if pd.unique(outer_groups).size < inner_splits:
            raise ValueError(
                f"Outer fold {outer_number} has too few groups for inner CV."
            )

        _require_class_group_support(
            outer_target,
            outer_groups,
            n_splits=inner_splits,
            context=f"outer fold {outer_number} inner CV",
        )

        inner_splitter = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + outer_number,
        )

        inner_placeholder = np.zeros(outer_train.size, dtype=np.uint8)
        inner_folds: list[GroupedFold] = []

        for inner_number, (
            inner_train_local,
            inner_validation_local,
        ) in enumerate(
            inner_splitter.split(
                inner_placeholder,
                outer_target,
                outer_groups,
            ),
            start=1,
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]

            assert_disjoint_groups(
                group_values[inner_train],
                group_values[inner_validation],
                context=(
                    f"outer fold {outer_number}, "
                    f"inner fold {inner_number}"
                ),
            )

            inner_folds.append(
                GroupedFold(
                    train_indices=inner_train,
                    validation_indices=inner_validation,
                )
            )

        nested_folds.append(
            NestedGroupedFold(
                number=outer_number,
                outer=GroupedFold(
                    train_indices=outer_train,
                    validation_indices=outer_validation,
                ),
                inner=tuple(inner_folds),
            )
        )

    return tuple(nested_folds)