"""Leakage-safe fitting and evaluation for registered development models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from anaemia_ml.evaluation import (
    MulticlassMetrics,
    align_probability_columns,
    assert_disjoint_groups,
    evaluate_multiclass,
)
from anaemia_ml.modeling.imbalance import build_primary_imbalance_classifier
from anaemia_ml.modeling.registry import build_estimator, model_spec
from anaemia_ml.preprocessing import build_model_pipeline

EvaluationScope = Literal["development_validation"]
DEFAULT_MODEL_NAMES = ("logistic_regression", "random_forest")


class RunnerError(ValueError):
    """Raised when a model run violates the development-only contract."""


@dataclass(frozen=True)
class ModelRun:
    """Auditable result from one held-out development validation split."""

    model_name: str
    display_name: str
    model_family: str
    variant: str
    scope: EvaluationScope
    random_seed: int
    training_rows: int
    validation_rows: int
    metrics: MulticlassMetrics
    pipeline: Pipeline

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe metadata and metrics without row-level data."""
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "model_family": self.model_family,
            "variant": self.variant,
            "scope": self.scope,
            "random_seed": self.random_seed,
            "training_rows": self.training_rows,
            "validation_rows": self.validation_rows,
            "metrics": self.metrics.as_dict(),
        }


def _validated_frame(
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise RunnerError(f"{name} must be a pandas DataFrame.")
    if frame.empty:
        raise RunnerError(f"{name} must not be empty.")
    return frame


def _validated_vector(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size != expected_length:
        raise RunnerError(
            f"{name} must be one-dimensional with {expected_length} values."
        )
    if bool(pd.isna(vector).any()):
        raise RunnerError(f"{name} must not contain missing values.")
    return vector


def _validated_weight(
    values: Sequence[float] | np.ndarray | None,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray | None:
    if values is None:
        return None
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{name} must contain numeric values.") from error
    if weights.shape != (expected_length,):
        raise RunnerError(f"{name} must contain one value per row.")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise RunnerError(
            f"{name} must be finite, non-negative, and have a positive sum."
        )
    return weights


def _validated_scope(scope: str) -> EvaluationScope:
    if scope != "development_validation":
        raise RunnerError(
            "This runner is development-only. Calibration, locked-test, and "
            "all-data evaluation must use their dedicated final workflows."
        )
    return "development_validation"


def _validation_seed(validation_config: Mapping[str, Any]) -> int:
    if not isinstance(validation_config, Mapping):
        raise RunnerError("validation_config must be a mapping.")
    value = validation_config.get("random_seed")
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise RunnerError(
            "validation_config.random_seed must be a non-negative integer."
        )
    return int(value)


def fit_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: Sequence[int] | np.ndarray,
    train_groups: Sequence[Any] | np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: Sequence[int] | np.ndarray,
    validation_groups: Sequence[Any] | np.ndarray,
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    scope: str = "development_validation",
    train_sample_weight: Sequence[float] | np.ndarray | None = None,
    validation_sample_weight: Sequence[float] | np.ndarray | None = None,
    n_jobs: int = -1,
    parameters: Mapping[str, Any] | None = None,
) -> ModelRun:
    """Fit one registered model and score one group-disjoint dev split."""
    checked_scope = _validated_scope(scope)
    train_frame = _validated_frame(X_train, name="X_train")
    validation_frame = _validated_frame(X_validation, name="X_validation")
    train_target = _validated_vector(
        y_train,
        name="y_train",
        expected_length=len(train_frame),
    )
    validation_target = _validated_vector(
        y_validation,
        name="y_validation",
        expected_length=len(validation_frame),
    )
    checked_train_groups = _validated_vector(
        train_groups,
        name="train_groups",
        expected_length=len(train_frame),
    )
    checked_validation_groups = _validated_vector(
        validation_groups,
        name="validation_groups",
        expected_length=len(validation_frame),
    )
    assert_disjoint_groups(
        checked_train_groups,
        checked_validation_groups,
        context="development validation",
    )

    checked_train_weight = _validated_weight(
        train_sample_weight,
        name="train_sample_weight",
        expected_length=len(train_frame),
    )
    checked_validation_weight = _validated_weight(
        validation_sample_weight,
        name="validation_sample_weight",
        expected_length=len(validation_frame),
    )
    if not isinstance(feature_schema, Mapping):
        raise RunnerError("feature_schema must be a mapping.")

    random_seed = _validation_seed(validation_config)
    spec = model_spec(model_name)
    estimator = build_estimator(
        model_name,
        random_seed=random_seed,
        n_jobs=n_jobs,
        parameters=parameters,
    )
    classifier = build_primary_imbalance_classifier(
        estimator,
        validation_config,
    )
    pipeline = build_model_pipeline(
        classifier,
        dict(feature_schema),
        variant=variant,
        model_family=spec.family,
        sparse_output=spec.sparse_output,
    )

    fit_parameters: dict[str, Any] = {}
    if checked_train_weight is not None:
        fit_parameters["model__sample_weight"] = checked_train_weight
    pipeline.fit(train_frame, train_target, **fit_parameters)

    predictions = pipeline.predict(validation_frame)
    fitted_classifier = pipeline.named_steps["model"]
    probabilities = align_probability_columns(
        pipeline.predict_proba(validation_frame),
        fitted_classifier.classes_,
    )
    metrics = evaluate_multiclass(
        validation_target,
        predictions,
        probabilities,
        sample_weight=checked_validation_weight,
    )

    return ModelRun(
        model_name=spec.name,
        display_name=spec.display_name,
        model_family=spec.family,
        variant=variant,
        scope=checked_scope,
        random_seed=random_seed,
        training_rows=len(train_frame),
        validation_rows=len(validation_frame),
        metrics=metrics,
        pipeline=pipeline,
    )


def compare_models(
    X_train: pd.DataFrame,
    y_train: Sequence[int] | np.ndarray,
    train_groups: Sequence[Any] | np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: Sequence[int] | np.ndarray,
    validation_groups: Sequence[Any] | np.ndarray,
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    model_parameters: Mapping[str, Mapping[str, Any]] | None = None,
    variant: str = "india_policy",
    scope: str = "development_validation",
    train_sample_weight: Sequence[float] | np.ndarray | None = None,
    validation_sample_weight: Sequence[float] | np.ndarray | None = None,
    n_jobs: int = -1,
) -> tuple[ModelRun, ...]:
    """Evaluate registered models in the caller's deterministic order."""
    if isinstance(model_names, (str, bytes)):
        raise RunnerError("model_names must be a sequence of model names.")
    names = tuple(model_names)
    if not names:
        raise RunnerError("model_names must not be empty.")
    if any(not isinstance(name, str) or not name for name in names):
        raise RunnerError("model_names must contain non-empty strings.")
    if len(names) != len(set(names)):
        raise RunnerError("model_names must not contain duplicates.")

    if model_parameters is None:
        parameter_map: Mapping[str, Mapping[str, Any]] = {}
    elif not isinstance(model_parameters, Mapping):
        raise RunnerError("model_parameters must be a mapping.")
    else:
        parameter_map = model_parameters

    if any(not isinstance(name, str) or not name for name in parameter_map):
        raise RunnerError("model_parameters keys must be non-empty model names.")
    if any(not isinstance(values, Mapping) for values in parameter_map.values()):
        raise RunnerError("Each model_parameters value must be a mapping.")

    unknown_parameter_models = sorted(set(parameter_map) - set(names))
    if unknown_parameter_models:
        raise RunnerError(
            "model_parameters contains models that were not requested: "
            f"{unknown_parameter_models}."
        )

    return tuple(
        fit_evaluate_model(
            name,
            X_train,
            y_train,
            train_groups,
            X_validation,
            y_validation,
            validation_groups,
            feature_schema,
            validation_config,
            variant=variant,
            scope=scope,
            train_sample_weight=train_sample_weight,
            validation_sample_weight=validation_sample_weight,
            n_jobs=n_jobs,
            parameters=parameter_map.get(name),
        )
        for name in names
    )
