"""Leakage-safe model-evaluation utilities."""

from anaemia_ml.evaluation.grouped_validation import (
    GroupedFold,
    GroupLeakageError,
    NestedGroupedFold,
    assert_disjoint_groups,
    nested_group_splits,
)

__all__ = [
    "GroupLeakageError",
    "GroupedFold",
    "NestedGroupedFold",
    "assert_disjoint_groups",
    "nested_group_splits",
]