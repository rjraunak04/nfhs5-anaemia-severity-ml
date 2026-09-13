"""Tests for the deterministic estimator registry."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from anaemia_ml.modeling.registry import (
    ModelRegistryError,
    build_estimator,
    model_spec,
    registered_models,
)


def test_registered_models_have_stable_order() -> None:
    assert tuple(spec.name for spec in registered_models()) == (
        "logistic_regression",
        "random_forest",
        "lightgbm",
    )


def test_registry_records_pipeline_family_and_optional_dependency() -> None:
    logistic = model_spec("logistic_regression")
    lightgbm = model_spec("lightgbm")

    assert logistic.family == "linear"
    assert logistic.is_optional is False
    assert lightgbm.family == "tree"
    assert lightgbm.optional_dependency == "lightgbm"
    assert lightgbm.is_optional is True


def test_logistic_regression_is_reproducible_and_unweighted() -> None:
    estimator = build_estimator("logistic_regression", random_seed=17)

    assert isinstance(estimator, LogisticRegression)
    assert estimator.random_state == 17
    assert estimator.class_weight is None
    assert estimator.max_iter == 2_000


def test_random_forest_uses_execution_controls() -> None:
    estimator = build_estimator(
        "random_forest",
        random_seed=23,
        n_jobs=2,
    )

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.random_state == 23
    assert estimator.n_jobs == 2
    assert estimator.class_weight is None


def test_safe_estimator_parameters_can_be_overridden() -> None:
    estimator = build_estimator(
        "random_forest",
        parameters={"n_estimators": 11, "max_depth": 4},
    )

    assert estimator.n_estimators == 11
    assert estimator.max_depth == 4


@pytest.mark.parametrize(
    "parameter",
    ["class_weight", "random_state", "n_jobs"],
)
def test_protocol_parameters_cannot_be_overridden(parameter: str) -> None:
    with pytest.raises(ModelRegistryError, match="cannot be overridden"):
        build_estimator(
            "random_forest",
            parameters={parameter: None},
        )


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ModelRegistryError, match="Unknown model"):
        build_estimator("xgboost")


@pytest.mark.parametrize("value", [True, -2, 0])
def test_invalid_n_jobs_is_rejected(value: int) -> None:
    with pytest.raises(ModelRegistryError, match="n_jobs"):
        build_estimator("random_forest", n_jobs=value)


def test_invalid_estimator_parameter_is_rejected() -> None:
    with pytest.raises(ModelRegistryError, match="Invalid parameters"):
        build_estimator(
            "logistic_regression",
            parameters={"not_a_parameter": 1},
        )


def test_non_string_parameter_name_is_rejected() -> None:
    with pytest.raises(ModelRegistryError, match="parameter names"):
        build_estimator(
            "logistic_regression",
            parameters={1: "invalid"},
        )
