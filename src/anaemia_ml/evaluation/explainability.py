"""Aggregate-only SHAP reporting for a held-out development partition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from anaemia_ml.preprocessing import predictor_groups, transformed_feature_names


class ExplainabilityError(ValueError):
    """Raised when an explainability report would be unsafe or inconsistent."""


def _positive_integer(value: Any, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= int(value) <= maximum:
        raise ExplainabilityError(f"{name} must be an integer in [1, {maximum}].")
    return int(value)


def _fitted_components(pipeline: Pipeline) -> tuple[Pipeline, Any, np.ndarray]:
    if not isinstance(pipeline, Pipeline):
        raise ExplainabilityError("pipeline must be a fitted sklearn Pipeline.")
    check_is_fitted(pipeline)
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["model"]
    except KeyError as error:
        raise ExplainabilityError("pipeline must contain preprocessor and model steps.") from error
    estimator = getattr(classifier, "estimator_", classifier)
    classes = np.asarray(getattr(estimator, "classes_", ()))
    if classes.ndim != 1 or classes.size < 2:
        raise ExplainabilityError("The fitted estimator must expose multiclass classes_.")
    return preprocessor, estimator, classes


def transformed_to_raw_features(
    preprocessor: Pipeline,
    feature_schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
) -> tuple[str, ...]:
    """Map imputed and one-hot output columns back to frozen raw predictors."""
    groups = predictor_groups(dict(feature_schema), variant=variant)
    try:
        columns = preprocessor.named_steps["columns"]
        numeric_pipeline = columns.named_transformers_["numeric"]
        categorical_pipeline = columns.named_transformers_["categorical"]
        imputer = numeric_pipeline.named_steps["imputer"]
        encoder = categorical_pipeline.named_steps["encoder"]
    except (AttributeError, KeyError) as error:
        raise ExplainabilityError("Unsupported fitted preprocessing structure.") from error

    raw_names = list(groups.numeric)
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        raw_names.extend(groups.numeric[int(index)] for index in indicator.features_)
    for column, categories in zip(groups.categorical, encoder.categories_, strict=True):
        raw_names.extend([column] * len(categories))

    output_names = transformed_feature_names(preprocessor)
    if len(raw_names) != len(output_names):
        raise ExplainabilityError(
            "Transformed feature map does not match preprocessing output width."
        )
    return tuple(raw_names)


def _normalized_shap_values(
    values: Any,
    *,
    sample_count: int,
    feature_count: int,
    class_count: int,
) -> np.ndarray:
    if isinstance(values, list):
        try:
            array = np.stack([np.asarray(value, dtype=float) for value in values], axis=2)
        except (TypeError, ValueError) as error:
            raise ExplainabilityError("SHAP returned inconsistent class arrays.") from error
    else:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ExplainabilityError("SHAP values must be numeric.") from error

    if array.shape == (sample_count, feature_count, class_count):
        normalized = array
    elif array.shape == (sample_count, class_count, feature_count):
        normalized = np.transpose(array, (0, 2, 1))
    elif class_count == 2 and array.shape == (sample_count, feature_count):
        normalized = np.stack([-array, array], axis=2)
    else:
        raise ExplainabilityError(
            f"Unexpected SHAP shape; expected samples x features x classes, received {array.shape}."
        )
    if not np.isfinite(normalized).all():
        raise ExplainabilityError("SHAP values must be finite.")
    return normalized


def _feature_rows(
    values: np.ndarray,
    raw_features: Sequence[str],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    unique = tuple(dict.fromkeys(raw_features))
    raw_array = np.asarray(raw_features, dtype=object)
    rows = []
    for feature in unique:
        selected = raw_array == feature
        feature_values = values[:, selected, :]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(feature_values))),
                "mean_signed_shap": float(np.mean(feature_values)),
            }
        )
    rows.sort(key=lambda row: (-row["mean_abs_shap"], row["feature"]))
    result = rows[:top_n]
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
    return result


def aggregate_shap_values(
    values: Any,
    raw_features: Sequence[str],
    classes: Sequence[int],
    *,
    sample_count: int,
    top_n: int = 20,
) -> dict[str, Any]:
    """Aggregate in-memory SHAP values without returning row-level arrays."""
    features = tuple(raw_features)
    labels = tuple(int(value) for value in classes)
    if not features or not labels:
        raise ExplainabilityError("raw_features and classes must not be empty.")
    checked_top_n = _positive_integer(top_n, name="top_n", maximum=len(set(features)))
    normalized = _normalized_shap_values(
        values,
        sample_count=sample_count,
        feature_count=len(features),
        class_count=len(labels),
    )
    per_class = []
    for class_index, label in enumerate(labels):
        class_values = normalized[:, :, class_index : class_index + 1]
        per_class.append(
            {
                "class_label": label,
                "features": _feature_rows(
                    class_values,
                    features,
                    top_n=checked_top_n,
                ),
            }
        )
    return {
        "global_features": _feature_rows(
            normalized,
            features,
            top_n=checked_top_n,
        ),
        "class_features": per_class,
    }


def _make_explainer(estimator: Any, background: np.ndarray) -> Any:
    try:
        shap = import_module("shap")
    except ImportError as error:
        raise ExplainabilityError(
            "SHAP is required for Day 2 explainability; install the modeling extra."
        ) from error
    if hasattr(estimator, "coef_"):
        return shap.LinearExplainer(estimator, background)
    if hasattr(estimator, "feature_importances_"):
        return shap.TreeExplainer(estimator)
    if not hasattr(estimator, "predict_proba"):
        raise ExplainabilityError("The fitted estimator does not support predict_proba.")
    return shap.Explainer(estimator.predict_proba, background)


def build_shap_report(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    feature_schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    partition_name: str = "calibration",
    max_samples: int = 200,
    background_size: int = 50,
    top_n: int = 20,
    random_state: int = 42,
    explainer_factory: Any | None = None,
) -> dict[str, Any]:
    """Compute and immediately aggregate SHAP on non-training observations."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ExplainabilityError("frame must be a non-empty pandas DataFrame.")
    if partition_name != "calibration":
        raise ExplainabilityError(
            "Day 2 SHAP is restricted to the calibration partition; locked test remains sealed."
        )
    sample_limit = _positive_integer(max_samples, name="max_samples", maximum=500)
    background_limit = _positive_integer(
        background_size,
        name="background_size",
        maximum=100,
    )
    preprocessor, estimator, classes = _fitted_components(pipeline)
    raw_features = transformed_to_raw_features(
        preprocessor,
        feature_schema,
        variant=variant,
    )
    rng = np.random.default_rng(int(random_state))
    selected_count = min(sample_limit, len(frame))
    selected_positions = np.sort(rng.choice(len(frame), size=selected_count, replace=False))
    selected = frame.iloc[selected_positions]
    transformed = preprocessor.transform(selected)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    if transformed.shape != (selected_count, len(raw_features)):
        raise ExplainabilityError("Transformed explanation matrix has an unexpected shape.")
    background_count = min(background_limit, selected_count)
    background_positions = np.linspace(
        0,
        selected_count - 1,
        num=background_count,
        dtype=int,
    )
    background = transformed[background_positions]
    factory = _make_explainer if explainer_factory is None else explainer_factory
    explainer = factory(estimator, background)
    explanation = explainer(transformed)
    shap_values = getattr(explanation, "values", explanation)
    aggregates = aggregate_shap_values(
        shap_values,
        raw_features,
        classes,
        sample_count=selected_count,
        top_n=top_n,
    )
    return {
        "method": "SHAP",
        "scope": "aggregate_only",
        "partition": partition_name,
        "sample_count": selected_count,
        "background_count": background_count,
        "variant": variant,
        "interpretation": "associational_not_causal",
        "model_stage": "development_candidate_before_temperature_scaling",
        "row_level_values_persisted": False,
        "locked_test_evaluated": False,
        **aggregates,
    }
