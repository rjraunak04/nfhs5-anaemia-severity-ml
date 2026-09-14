"""Leakage-safe model-evaluation utilities."""

from anaemia_ml.evaluation.calibration import (
    DEFAULT_TEMPERATURES,
    CalibrationError,
    CalibrationSelection,
    TemperatureScaledClassifier,
    apply_temperature,
    select_temperature_scaling,
)
from anaemia_ml.evaluation.config import (
    FINAL_TEST_UNLOCK_TOKEN,
    FinalTestLockedError,
    ValidationConfigError,
    load_validation_config,
    require_final_test_unlock,
    validate_validation_config,
)
from anaemia_ml.evaluation.explainability import (
    ExplainabilityError,
    aggregate_shap_values,
    build_shap_report,
    transformed_to_raw_features,
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
    "DEFAULT_TEMPERATURES",
    "FINAL_TEST_UNLOCK_TOKEN",
    "SEVERE_CLASS",
    "CalibrationError",
    "CalibrationSelection",
    "ExplainabilityError",
    "FinalTestLockedError",
    "GroupLeakageError",
    "GroupedFold",
    "GroupedPartitions",
    "MetricError",
    "MulticlassMetrics",
    "NestedGroupedFold",
    "PartitionError",
    "TemperatureScaledClassifier",
    "ValidationConfigError",
    "aggregate_shap_values",
    "align_probability_columns",
    "apply_temperature",
    "assert_disjoint_groups",
    "assert_group_disjoint_partitions",
    "build_shap_report",
    "evaluate_multiclass",
    "expected_calibration_error",
    "load_validation_config",
    "make_grouped_partitions",
    "multiclass_brier_score",
    "nested_group_splits",
    "ordinal_mean_absolute_error",
    "require_final_test_unlock",
    "select_temperature_scaling",
    "transformed_to_raw_features",
    "validate_validation_config",
]
