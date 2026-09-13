"""Leakage-safe model-evaluation utilities."""

from anaemia_ml.evaluation.config import (
    FINAL_TEST_UNLOCK_TOKEN,
    FinalTestLockedError,
    ValidationConfigError,
    load_validation_config,
    require_final_test_unlock,
    validate_validation_config,
)
from anaemia_ml.evaluation.grouped_validation import (
    GroupedFold,
    GroupLeakageError,
    NestedGroupedFold,
    assert_disjoint_groups,
    nested_group_splits,
)
from anaemia_ml.evaluation.metrics import (
    DEFAULT_CLASSES,
    SEVERE_CLASS,
    MetricError,
    MulticlassMetrics,
    align_probability_columns,
    evaluate_multiclass,
    expected_calibration_error,
    multiclass_brier_score,
    ordinal_mean_absolute_error,
)
from anaemia_ml.evaluation.partitions import (
    GroupedPartitions,
    PartitionError,
    assert_group_disjoint_partitions,
    make_grouped_partitions,
)

__all__ = [
    "DEFAULT_CLASSES",
    "FINAL_TEST_UNLOCK_TOKEN",
    "SEVERE_CLASS",
    "FinalTestLockedError",
    "GroupedFold",
    "GroupedPartitions",
    "GroupLeakageError",
    "MetricError",
    "MulticlassMetrics",
    "NestedGroupedFold",
    "PartitionError",
    "ValidationConfigError",
    "align_probability_columns",
    "assert_disjoint_groups",
    "assert_group_disjoint_partitions",
    "evaluate_multiclass",
    "expected_calibration_error",
    "load_validation_config",
    "make_grouped_partitions",
    "multiclass_brier_score",
    "nested_group_splits",
    "ordinal_mean_absolute_error",
    "require_final_test_unlock",
    "validate_validation_config",
]
