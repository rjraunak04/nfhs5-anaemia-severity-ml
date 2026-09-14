"""Group-safe temperature scaling for the held-out calibration partition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from anaemia_ml.evaluation.metrics import (
    DEFAULT_CLASSES,
    MulticlassMetrics,
    align_probability_columns,
    evaluate_multiclass,
)

DEFAULT_TEMPERATURES = (
    0.50,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.50,
    1.75,
    2.0,
    2.5,
    3.0,
)


class CalibrationError(ValueError):
    """Raised when calibration would be invalid or leak group information."""


@dataclass(frozen=True)
class CalibrationSelection:
    """Aggregate calibration-selection result; contains no row-level values."""

    method: str
    temperature: float
    candidate_temperature: float
    accepted: bool
    cross_fit_splits: int
    before: MulticlassMetrics
    cross_fitted_candidate: MulticlassMetrics

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe summary."""
        deployed = self.cross_fitted_candidate if self.accepted else self.before
        return {
            "method": self.method,
            "selection_objective": "survey_weighted_log_loss",
            "temperature": float(self.temperature),
            "candidate_temperature": float(self.candidate_temperature),
            "accepted": self.accepted,
            "cross_fit_splits": self.cross_fit_splits,
            "group_disjoint_cross_fit": True,
            "metrics_before": self.before.as_dict(),
            "candidate_metrics_cross_fitted": self.cross_fitted_candidate.as_dict(),
            "deployed_metrics_estimate": deployed.as_dict(),
            "evaluation_scope": "calibration_selection_only",
            "locked_test_evaluated": False,
        }


def apply_temperature(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply multiclass temperature scaling without changing class order."""
    try:
        values = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationError("probabilities must be numeric.") from error
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise CalibrationError(
            "probabilities must be a non-empty two-dimensional array."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise CalibrationError("probabilities must be finite and non-negative.")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise CalibrationError("Every probability row must sum to 1.")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise CalibrationError("temperature must be numeric.")
    checked_temperature = float(temperature)
    if not np.isfinite(checked_temperature) or checked_temperature <= 0:
        raise CalibrationError("temperature must be finite and positive.")

    logits = np.log(np.clip(values, 1e-15, 1.0)) / checked_temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _validated_vector(values: Any, *, name: str, sample_count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (sample_count,):
        raise CalibrationError(f"{name} must contain one value per calibration row.")
    return result


def _validated_weights(values: Any, *, sample_count: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationError("sample_weight must be numeric.") from error
    if weights.shape != (sample_count,):
        raise CalibrationError(
            "sample_weight must contain one value per calibration row."
        )
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise CalibrationError(
            "sample_weight must be finite, non-negative, and non-zero."
        )
    return weights


def _validated_temperatures(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise CalibrationError("temperatures must be a numeric sequence.")
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CalibrationError("temperatures must contain numeric values.")
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise CalibrationError("temperatures must be finite and positive.")
        checked.append(number)
    if not checked:
        raise CalibrationError("temperatures must not be empty.")
    if 1.0 not in checked:
        checked.append(1.0)
    return tuple(sorted(set(checked)))


def _best_temperature(
    probabilities: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None,
    classes: tuple[int, ...],
    temperatures: tuple[float, ...],
) -> float:
    scored = []
    for temperature in temperatures:
        scaled = apply_temperature(probabilities, temperature)
        score = float(
            log_loss(
                target,
                scaled,
                labels=list(classes),
                sample_weight=sample_weight,
            )
        )
        scored.append((score, abs(np.log(temperature)), temperature))
    return float(min(scored)[2])


def _feasible_split_count(
    target: np.ndarray,
    groups: np.ndarray,
    classes: tuple[int, ...],
    requested_splits: int,
) -> int:
    if (
        isinstance(requested_splits, bool)
        or not isinstance(requested_splits, Integral)
        or int(requested_splits) < 2
    ):
        raise CalibrationError("n_splits must be an integer >= 2.")
    unique_groups = np.unique(groups)
    groups_per_class = [np.unique(groups[target == label]).size for label in classes]
    split_count = min(int(requested_splits), unique_groups.size, *groups_per_class)
    if split_count < 2:
        raise CalibrationError(
            "Calibration needs at least two distinct PSU groups for every class."
        )
    return int(split_count)


def select_temperature_scaling(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    model_classes: Sequence[int] = DEFAULT_CLASSES,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    n_splits: int = 5,
    random_state: int = 42,
) -> CalibrationSelection:
    """Select temperature scaling with PSU-disjoint cross-fitting.

    The development model is already frozen. Only its calibration-partition
    probabilities, labels, groups and survey weights are used here. A candidate
    is deployed only when its cross-fitted weighted log loss beats identity.
    """
    raw_classes = tuple(expected_classes)
    if not raw_classes or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in raw_classes
    ):
        raise CalibrationError("expected_classes must contain integers.")
    classes = tuple(int(value) for value in raw_classes)
    if len(classes) != len(set(classes)):
        raise CalibrationError("expected_classes must be unique.")
    try:
        raw = np.asarray(y_proba, dtype=float)
    except (TypeError, ValueError) as error:
        raise CalibrationError("y_proba must be numeric.") from error
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise CalibrationError("y_proba must be a non-empty two-dimensional array.")
    probabilities = align_probability_columns(
        raw,
        model_classes,
        expected_classes=classes,
    )
    raw_target = _validated_vector(y_true, name="y_true", sample_count=raw.shape[0])
    try:
        numeric_target = raw_target.astype(float)
    except (TypeError, ValueError) as error:
        raise CalibrationError("y_true must contain integer class labels.") from error
    if (
        not np.isfinite(numeric_target).all()
        or not np.equal(
            numeric_target,
            np.floor(numeric_target),
        ).all()
    ):
        raise CalibrationError("y_true must contain finite integer class labels.")
    target = numeric_target.astype(int)
    if set(target.tolist()) != set(classes):
        raise CalibrationError("Calibration data must contain every expected class.")
    group_values = _validated_vector(
        groups,
        name="groups",
        sample_count=raw.shape[0],
    )
    for value in group_values.tolist():
        missing_number = isinstance(value, (float, np.floating)) and not np.isfinite(
            value
        )
        if (
            value is None
            or missing_number
            or (isinstance(value, str) and not value.strip())
        ):
            raise CalibrationError("groups must not contain missing or empty values.")
    weights = _validated_weights(sample_weight, sample_count=raw.shape[0])
    candidates = _validated_temperatures(temperatures)
    split_count = _feasible_split_count(
        target,
        group_values,
        classes,
        n_splits,
    )

    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=int(random_state),
    )
    cross_fitted = np.empty_like(probabilities)
    assigned = np.zeros(raw.shape[0], dtype=bool)
    for train_index, validation_index in splitter.split(
        probabilities,
        target,
        group_values,
    ):
        overlap = set(group_values[train_index]) & set(group_values[validation_index])
        if overlap:
            raise CalibrationError(
                "PSU groups overlap inside calibration cross-fitting."
            )
        fold_weights = None if weights is None else weights[train_index]
        temperature = _best_temperature(
            probabilities[train_index],
            target[train_index],
            fold_weights,
            classes,
            candidates,
        )
        cross_fitted[validation_index] = apply_temperature(
            probabilities[validation_index],
            temperature,
        )
        assigned[validation_index] = True
    if not assigned.all():
        raise CalibrationError("Cross-fitting did not evaluate every calibration row.")

    labels = np.asarray(classes)
    before = evaluate_multiclass(
        target,
        labels[np.argmax(probabilities, axis=1)],
        probabilities,
        classes=classes,
        sample_weight=weights,
    )
    candidate = evaluate_multiclass(
        target,
        labels[np.argmax(cross_fitted, axis=1)],
        cross_fitted,
        classes=classes,
        sample_weight=weights,
    )
    candidate_temperature = _best_temperature(
        probabilities,
        target,
        weights,
        classes,
        candidates,
    )
    accepted = candidate.log_loss < before.log_loss - 1e-12
    return CalibrationSelection(
        method="temperature_scaling" if accepted else "identity",
        temperature=candidate_temperature if accepted else 1.0,
        candidate_temperature=candidate_temperature,
        accepted=accepted,
        cross_fit_splits=split_count,
        before=before,
        cross_fitted_candidate=candidate,
    )


class TemperatureScaledClassifier(ClassifierMixin, BaseEstimator):
    """Prediction-only temperature wrapper around a trusted fitted classifier."""

    def __init__(
        self,
        fitted_estimator: Any,
        *,
        temperature: float,
        classes: Sequence[int] = DEFAULT_CLASSES,
    ) -> None:
        self.fitted_estimator = fitted_estimator
        self.temperature = temperature
        self.classes = classes
        self.classes_ = np.asarray(tuple(classes), dtype=int)

    def __sklearn_is_fitted__(self) -> bool:
        """Mark the prediction-only wrapper as fitted for sklearn validation."""
        return hasattr(self.fitted_estimator, "predict_proba")

    def fit(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> TemperatureScaledClassifier:
        """Reject refitting because the wrapped development model is frozen."""
        del X, y, fit_params
        raise CalibrationError(
            "TemperatureScaledClassifier wraps a frozen fitted model and cannot be refit."
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class-aligned temperature-scaled probabilities."""
        raw = self.fitted_estimator.predict_proba(X)
        model_classes = getattr(self.fitted_estimator, "classes_", self.classes_)
        aligned = align_probability_columns(
            raw,
            model_classes,
            expected_classes=self.classes_,
        )
        return apply_temperature(aligned, self.temperature)

    def predict(self, X: Any) -> np.ndarray:
        """Return labels in the frozen class order."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
