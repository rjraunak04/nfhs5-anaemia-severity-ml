"""Tests for leakage-safe three-way grouped partitions."""

import numpy as np
import pytest

from anaemia_ml.evaluation.grouped_validation import GroupLeakageError
from anaemia_ml.evaluation.partitions import (
    GroupedPartitions,
    PartitionError,
    assert_group_disjoint_partitions,
    make_grouped_partitions,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Create balanced four-class data distributed across 80 PSUs."""
    groups = np.repeat(np.arange(80), 12)
    y = np.tile(np.arange(4), 240)

    order = np.random.default_rng(2026).permutation(y.size)
    return y[order], groups[order]


def test_partition_sizes_follow_protocol(synthetic_data) -> None:
    y, groups = synthetic_data
    partitions = make_grouped_partitions(y, groups)

    assert partitions.development.size / y.size == pytest.approx(
        0.70,
        abs=0.02,
    )
    assert partitions.calibration.size / y.size == pytest.approx(
        0.10,
        abs=0.02,
    )
    assert partitions.locked_test.size / y.size == pytest.approx(
        0.20,
        abs=0.02,
    )


def test_partitions_cover_each_row_once(synthetic_data) -> None:
    y, groups = synthetic_data
    partitions = make_grouped_partitions(y, groups)

    combined = np.concatenate(
        tuple(partitions.as_dict().values())
    )

    np.testing.assert_array_equal(
        np.sort(combined),
        np.arange(y.size),
    )


def test_psu_groups_are_disjoint(synthetic_data) -> None:
    y, groups = synthetic_data
    partitions = make_grouped_partitions(y, groups)

    group_sets = {
        name: set(groups[indices])
        for name, indices in partitions.as_dict().items()
    }

    assert group_sets["development"].isdisjoint(
        group_sets["calibration"]
    )
    assert group_sets["development"].isdisjoint(
        group_sets["locked_test"]
    )
    assert group_sets["calibration"].isdisjoint(
        group_sets["locked_test"]
    )


def test_partitioning_is_deterministic(synthetic_data) -> None:
    y, groups = synthetic_data

    first = make_grouped_partitions(
        y,
        groups,
        random_state=42,
    )
    second = make_grouped_partitions(
        y,
        groups,
        random_state=42,
    )

    for name in first.as_dict():
        np.testing.assert_array_equal(
            first.as_dict()[name],
            second.as_dict()[name],
        )


def test_each_partition_retains_all_classes(
    synthetic_data,
) -> None:
    y, groups = synthetic_data
    partitions = make_grouped_partitions(y, groups)
    expected_classes = set(np.unique(y))

    for indices in partitions.as_dict().values():
        assert set(np.unique(y[indices])) == expected_classes


def test_mismatched_input_lengths_are_rejected() -> None:
    with pytest.raises(
        PartitionError,
        match="same number of rows",
    ):
        make_grouped_partitions(
            np.array([0, 1, 0]),
            np.array([1, 2]),
        )


def test_invalid_fraction_sum_is_rejected(
    synthetic_data,
) -> None:
    y, groups = synthetic_data

    with pytest.raises(
        PartitionError,
        match="sum to 1.0",
    ):
        make_grouped_partitions(
            y,
            groups,
            development_fraction=0.60,
            calibration_fraction=0.10,
            locked_test_fraction=0.20,
        )


def test_incompatible_fraction_is_rejected(
    synthetic_data,
) -> None:
    y, groups = synthetic_data

    with pytest.raises(
        PartitionError,
        match="reciprocal",
    ):
        make_grouped_partitions(
            y,
            groups,
            development_fraction=0.65,
            calibration_fraction=0.15,
            locked_test_fraction=0.20,
        )


def test_single_class_outcome_is_rejected() -> None:
    y = np.zeros(100, dtype=int)
    groups = np.repeat(np.arange(20), 5)

    with pytest.raises(
        PartitionError,
        match="at least two classes",
    ):
        make_grouped_partitions(y, groups)


def test_overlap_checker_detects_group_leakage() -> None:
    partitions = GroupedPartitions(
        development=np.array([0, 1]),
        calibration=np.array([2]),
        locked_test=np.array([3]),
    )
    groups = np.array([10, 11, 10, 12])

    with pytest.raises(
        GroupLeakageError,
        match="Group leakage",
    ):
        assert_group_disjoint_partitions(
            partitions,
            groups,
        )