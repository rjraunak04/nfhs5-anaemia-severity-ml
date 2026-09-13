"""Tests for leakage-safe registered model runs."""

import json

import numpy as np
import pandas as pd
import pytest

from anaemia_ml.evaluation.grouped_validation import GroupLeakageError
from anaemia_ml.modeling.runner import (
    RunnerError,
    compare_models,
    fit_evaluate_model,
)


@pytest.fixture
def feature_schema() -> dict:
    return {
        "feature_groups": {
            "continuous_numeric": ["age", "bmi"],
            "count_numeric": ["children"],
            "nominal_categorical": ["region"],
            "indicator_categorical": ["rural"],
            "ordinal_categorical": ["wealth"],
        },
        "model_variants": {
            "india_policy": {
                "excluded_features": [],
                "include_groups": [
                    "continuous_numeric",
                    "count_numeric",
                    "nominal_categorical",
                    "indicator_categorical",
                    "ordinal_categorical",
                ],
            }
        },
        "preprocessing_contract": {
            "fit_scope": "training_fold_only",
            "numeric": {
                "imputation": "median",
                "add_missing_indicator": True,
                "scaling": "model_dependent",
            },
            "categorical": {
                "imputation": "explicit_missing_category",
                "unknown_category_policy": "handle_without_failure",
                "linear_model_encoding": "one_hot",
                "tree_model_encoding": "model_appropriate",
                "verify_ordinal_order_from_official_labels": True,
            },
            "engineered_features": {
                "primary_analysis": [],
                "post_hoc_features_allowed": False,
            },
        },
        "leakage_guards": {
            "forbidden_predictors": ["haemoglobin"],
            "preprocessing_before_split_allowed": False,
        },
    }


@pytest.fixture
def validation_config() -> dict:
    return {
        "random_seed": 42,
        "preprocessing": {
            "fit_inside_training_fold_only": True,
            "primary_imbalance_strategy": "class_weight",
            "smotenc_sensitivity_ratio": 0.45,
        },
    }


def _frame(labels: np.ndarray, *, offset: int) -> pd.DataFrame:
    row = np.arange(labels.size) + offset
    return pd.DataFrame(
        {
            "age": 20.0 + labels * 8 + row % 3,
            "bmi": 18.0 + labels * 2 + row % 2,
            "children": labels + row % 2,
            "region": np.asarray(["north", "south", "east", "west"])[labels],
            "rural": np.asarray(["no", "yes", "yes", "no"])[labels],
            "wealth": np.asarray(["low", "middle", "high", "high"])[labels],
            "ignored_id": row,
        }
    )


@pytest.fixture
def split_data() -> tuple:
    train_target = np.repeat(np.arange(4), 10)
    validation_target = np.repeat(np.arange(4), 4)
    return (
        _frame(train_target, offset=0),
        train_target,
        np.repeat(np.arange(20), 2),
        _frame(validation_target, offset=100),
        validation_target,
        np.repeat(np.arange(100, 108), 2),
    )


def test_logistic_model_run_is_complete_and_json_safe(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    result = fit_evaluate_model(
        "logistic_regression",
        *split_data,
        feature_schema,
        validation_config,
    )

    assert result.model_name == "logistic_regression"
    assert result.model_family == "linear"
    assert result.training_rows == 40
    assert result.validation_rows == 16
    assert result.scope == "development_validation"
    assert result.metrics.macro_f1 >= 0.0
    assert result.pipeline.named_steps["model"].class_weight_ == {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
    }
    json.dumps(result.summary(), allow_nan=False)


def test_same_seed_produces_same_predictions_and_metrics(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    first = fit_evaluate_model(
        "random_forest",
        *split_data,
        feature_schema,
        validation_config,
        n_jobs=1,
        parameters={"n_estimators": 12},
    )
    second = fit_evaluate_model(
        "random_forest",
        *split_data,
        feature_schema,
        validation_config,
        n_jobs=1,
        parameters={"n_estimators": 12},
    )

    np.testing.assert_array_equal(
        first.pipeline.predict(split_data[3]),
        second.pipeline.predict(split_data[3]),
    )
    assert first.metrics.as_dict() == pytest.approx(second.metrics.as_dict())


def test_overlapping_groups_are_rejected(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    values = list(split_data)
    values[5] = np.repeat(np.arange(8), 2)

    with pytest.raises(GroupLeakageError, match="overlapping groups"):
        fit_evaluate_model(
            "logistic_regression",
            *values,
            feature_schema,
            validation_config,
        )


@pytest.mark.parametrize("scope", ["calibration", "locked_test", "all_data"])
def test_non_development_scope_is_rejected(
    scope: str,
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="development-only"):
        fit_evaluate_model(
            "logistic_regression",
            *split_data,
            feature_schema,
            validation_config,
            scope=scope,
        )


def test_dataframe_inputs_are_required(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="pandas DataFrame"):
        fit_evaluate_model(
            "logistic_regression",
            split_data[0].to_numpy(),
            *split_data[1:],
            feature_schema,
            validation_config,
        )


def test_train_and_validation_weights_are_accepted(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    result = fit_evaluate_model(
        "logistic_regression",
        *split_data,
        feature_schema,
        validation_config,
        train_sample_weight=np.linspace(0.5, 1.5, 40),
        validation_sample_weight=np.linspace(0.5, 1.5, 16),
    )

    assert np.isfinite(list(result.metrics.as_dict().values())).all()


def test_compare_models_preserves_requested_order(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    results = compare_models(
        *split_data,
        feature_schema,
        validation_config,
        model_names=("random_forest", "logistic_regression"),
        model_parameters={"random_forest": {"n_estimators": 8}},
        n_jobs=1,
    )

    assert tuple(result.model_name for result in results) == (
        "random_forest",
        "logistic_regression",
    )


def test_compare_models_rejects_duplicates(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="duplicates"):
        compare_models(
            *split_data,
            feature_schema,
            validation_config,
            model_names=("random_forest", "random_forest"),
        )


def test_compare_models_rejects_plain_string_names(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="sequence of model names"):
        compare_models(
            *split_data,
            feature_schema,
            validation_config,
            model_names="random_forest",
        )


def test_compare_models_rejects_unrequested_parameter_map(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="were not requested"):
        compare_models(
            *split_data,
            feature_schema,
            validation_config,
            model_names=("logistic_regression",),
            model_parameters={"random_forest": {"n_estimators": 8}},
        )


def test_compare_models_rejects_non_mapping_parameters(
    split_data: tuple,
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(RunnerError, match="value must be a mapping"):
        compare_models(
            *split_data,
            feature_schema,
            validation_config,
            model_names=("logistic_regression",),
            model_parameters={"logistic_regression": None},
        )
