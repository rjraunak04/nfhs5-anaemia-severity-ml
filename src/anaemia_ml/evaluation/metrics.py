"""Validated metrics for four-class anaemia severity evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_CLASSES = (0, 1, 2, 3)
SEVERE_CLASS = 3


class MetricError(ValueError):
    """Raised when metric inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class MulticlassMetrics:
    """Protocol metrics from one held-out evaluation partition."""

    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    macro_roc_auc_ovr: float
    severe_recall: float
    severe_precision: float
    severe_pr_auc: float
    multiclass_brier: float
    log_loss: float
    expected_calibration_error: float
    ordinal_mae: float
    quadratic_weighted_kappa: float

    def as_dict(self) -> dict[str, float]:
        """Return a stable, JSON-serializable metric mapping."""
        return {
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_roc_auc_ovr": float(self.macro_roc_auc_ovr),
            "severe_recall": float(self.severe_recall),
            "severe_precision": float(self.severe_precision),
            "severe_pr_auc": float(self.severe_pr_auc),
            "multiclass_brier": float(self.multiclass_brier),
            "log_loss": float(self.log_loss),
            "expected_calibration_error": float(self.expected_calibration_error),
            "ordinal_mae": float(self.ordinal_mae),
            "quadratic_weighted_kappa": float(self.quadratic_weighted_kappa),
        }


def _validated_classes(
    values: Sequence[int],
    *,
    name: str = "classes",
) -> tuple[int, ...]:
    raw = tuple(values)

    if not raw:
        raise MetricError(f"{name} must not be empty.")

    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in raw
    ):
        raise MetricError(f"{name} must contain integers only.")

    classes = tuple(int(value) for value in raw)

    if len(classes) != len(set(classes)):
        raise MetricError(f"{name} must not contain duplicates.")

    return classes


def _validated_label_vector(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    classes: tuple[int, ...],
    require_all_classes: bool,
) -> np.ndarray:
    labels = np.asarray(values)

    if labels.ndim != 1 or labels.size == 0:
        raise MetricError(f"{name} must be a non-empty one-dimensional sequence.")

    if np.issubdtype(labels.dtype, np.bool_):
        raise MetricError(f"{name} must contain integer class labels.")

    try:
        numeric = labels.astype(float)
    except (TypeError, ValueError) as error:
        raise MetricError(f"{name} must contain integer class labels.") from error

    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise MetricError(f"{name} must contain finite integer class labels.")

    integer_labels = numeric.astype(int)
    observed = set(integer_labels.tolist())
    unexpected = sorted(observed - set(classes))

    if unexpected:
        raise MetricError(f"Unexpected {name} classes: {unexpected}")

    if require_all_classes:
        missing = sorted(set(classes) - observed)
        if missing:
            raise MetricError(f"{name} is missing expected classes: {missing}")

    return integer_labels


def _validated_probabilities(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    sample_count: int,
    class_count: int,
) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise MetricError("y_proba must contain numeric values.") from error

    expected_shape = (sample_count, class_count)
    if probabilities.shape != expected_shape:
        raise MetricError(
            f"y_proba must have shape {expected_shape}; received {probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise MetricError("y_proba must contain finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise MetricError("y_proba values must lie in [0, 1].")

    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise MetricError("Every y_proba row must sum to 1.")

    return probabilities


def _validated_sample_weight(
    sample_weight: Sequence[float] | np.ndarray | None,
    *,
    sample_count: int,
) -> np.ndarray | None:
    if sample_weight is None:
        return None

    try:
        weights = np.asarray(sample_weight, dtype=float)
    except (TypeError, ValueError) as error:
        raise MetricError("sample_weight must contain numeric values.") from error

    if weights.shape != (sample_count,):
        raise MetricError("sample_weight must contain one value per evaluation row.")

    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise MetricError("sample_weight must be finite, non-negative, and non-zero.")

    return weights


def _validated_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or int(n_bins) < 2:
        raise MetricError("n_bins must be an integer >= 2.")

    return int(n_bins)


def align_probability_columns(
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    model_classes: Sequence[int],
    *,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
) -> np.ndarray:
    """Reorder model probability columns to the protocol class order."""
    expected = _validated_classes(
        expected_classes,
        name="expected_classes",
    )
    observed = _validated_classes(
        model_classes,
        name="model_classes",
    )

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise MetricError(
            "model_classes must match expected_classes exactly; "
            f"missing={missing}, unexpected={unexpected}."
        )

    try:
        raw = np.asarray(y_proba, dtype=float)
    except (TypeError, ValueError) as error:
        raise MetricError("y_proba must contain numeric values.") from error

    if raw.ndim != 2 or raw.shape[0] == 0:
        raise MetricError("y_proba must be a non-empty two-dimensional array.")

    probabilities = _validated_probabilities(
        y_proba,
        sample_count=raw.shape[0],
        class_count=len(observed),
    )
    positions = {label: index for index, label in enumerate(observed)}

    return probabilities[:, [positions[label] for label in expected]]


def multiclass_brier_score(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[int] = DEFAULT_CLASSES,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return the unscaled multiclass Brier score (lower is better)."""
    expected = _validated_classes(classes)
    true_labels = _validated_label_vector(
        y_true,
        name="y_true",
        classes=expected,
        require_all_classes=False,
    )
    probabilities = _validated_probabilities(
        y_proba,
        sample_count=true_labels.size,
        class_count=len(expected),
    )
    weights = _validated_sample_weight(
        sample_weight,
        sample_count=true_labels.size,
    )

    one_hot = (
        true_labels[:, np.newaxis] == np.asarray(expected)[np.newaxis, :]
    ).astype(float)
    row_scores = np.square(probabilities - one_hot).sum(axis=1)

    return float(np.average(row_scores, weights=weights))


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[int] = DEFAULT_CLASSES,
    n_bins: int = 10,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return equal-width top-label expected calibration error."""
    expected = _validated_classes(classes)
    bin_count = _validated_bin_count(n_bins)
    true_labels = _validated_label_vector(
        y_true,
        name="y_true",
        classes=expected,
        require_all_classes=False,
    )
    probabilities = _validated_probabilities(
        y_proba,
        sample_count=true_labels.size,
        class_count=len(expected),
    )
    weights = _validated_sample_weight(
        sample_weight,
        sample_count=true_labels.size,
    )
    effective_weights = (
        np.ones(true_labels.size, dtype=float) if weights is None else weights
    )

    predicted_positions = np.argmax(probabilities, axis=1)
    predicted_labels = np.asarray(expected)[predicted_positions]
    confidence = probabilities[
        np.arange(true_labels.size),
        predicted_positions,
    ]
    correctness = (predicted_labels == true_labels).astype(float)
    bin_indices = np.minimum(
        np.floor(confidence * bin_count).astype(int),
        bin_count - 1,
    )

    total_weight = float(effective_weights.sum())
    calibration_error = 0.0

    for bin_index in range(bin_count):
        selected = bin_indices == bin_index
        if not selected.any():
            continue

        bin_weights = effective_weights[selected]
        bin_weight = float(bin_weights.sum())
        if bin_weight <= 0:
            continue

        bin_accuracy = float(np.average(correctness[selected], weights=bin_weights))
        bin_confidence = float(np.average(confidence[selected], weights=bin_weights))
        calibration_error += (
            bin_weight / total_weight * abs(bin_accuracy - bin_confidence)
        )

    return float(calibration_error)


def ordinal_mean_absolute_error(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    classes: Sequence[int] = DEFAULT_CLASSES,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return mean absolute distance between ordered class positions."""
    expected = _validated_classes(classes)
    true_labels = _validated_label_vector(
        y_true,
        name="y_true",
        classes=expected,
        require_all_classes=False,
    )
    predicted_labels = _validated_label_vector(
        y_pred,
        name="y_pred",
        classes=expected,
        require_all_classes=False,
    )

    if true_labels.size != predicted_labels.size:
        raise MetricError("y_true and y_pred must have the same length.")

    weights = _validated_sample_weight(
        sample_weight,
        sample_count=true_labels.size,
    )
    rank = {label: index for index, label in enumerate(expected)}
    true_ranks = np.fromiter(
        (rank[int(label)] for label in true_labels),
        dtype=int,
        count=true_labels.size,
    )
    predicted_ranks = np.fromiter(
        (rank[int(label)] for label in predicted_labels),
        dtype=int,
        count=predicted_labels.size,
    )

    return float(
        np.average(
            np.abs(true_ranks - predicted_ranks),
            weights=weights,
        )
    )


def evaluate_multiclass(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence[int] = DEFAULT_CLASSES,
    severe_class: int = SEVERE_CLASS,
    n_bins: int = 10,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> MulticlassMetrics:
    """Compute frozen classification, calibration, and ordinal metrics.

    ``severe_pr_auc`` uses average precision, the non-interpolated summary of
    the severe-versus-rest precision-recall curve.
    """
    expected = _validated_classes(classes)
    if (
        isinstance(severe_class, (bool, np.bool_))
        or not isinstance(severe_class, Integral)
        or int(severe_class) not in expected
    ):
        raise MetricError("severe_class must be one of classes.")

    severe_label = int(severe_class)
    bin_count = _validated_bin_count(n_bins)
    true_labels = _validated_label_vector(
        y_true,
        name="y_true",
        classes=expected,
        require_all_classes=True,
    )
    predicted_labels = _validated_label_vector(
        y_pred,
        name="y_pred",
        classes=expected,
        require_all_classes=False,
    )

    if true_labels.size != predicted_labels.size:
        raise MetricError("y_true and y_pred must have the same length.")

    probabilities = _validated_probabilities(
        y_proba,
        sample_count=true_labels.size,
        class_count=len(expected),
    )
    weights = _validated_sample_weight(
        sample_weight,
        sample_count=true_labels.size,
    )

    if weights is not None:
        zero_weight_classes = [
            label
            for label in expected
            if float(weights[true_labels == label].sum()) <= 0
        ]
        if zero_weight_classes:
            raise MetricError(
                "Every y_true class must have positive total sample weight; "
                f"zero-weight classes: {zero_weight_classes}."
            )

    true_severe = true_labels == severe_label
    predicted_severe = predicted_labels == severe_label
    severe_position = expected.index(severe_label)

    return MulticlassMetrics(
        macro_f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=list(expected),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=list(expected),
                average="weighted",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
                sample_weight=weights,
            )
        ),
        macro_roc_auc_ovr=float(
            roc_auc_score(
                true_labels,
                probabilities,
                labels=list(expected),
                multi_class="ovr",
                average="macro",
                sample_weight=weights,
            )
        ),
        severe_recall=float(
            recall_score(
                true_severe,
                predicted_severe,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        severe_precision=float(
            precision_score(
                true_severe,
                predicted_severe,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        severe_pr_auc=float(
            average_precision_score(
                true_severe,
                probabilities[:, severe_position],
                sample_weight=weights,
            )
        ),
        multiclass_brier=multiclass_brier_score(
            true_labels,
            probabilities,
            classes=expected,
            sample_weight=weights,
        ),
        log_loss=float(
            log_loss(
                true_labels,
                probabilities,
                labels=list(expected),
                sample_weight=weights,
            )
        ),
        expected_calibration_error=expected_calibration_error(
            true_labels,
            probabilities,
            classes=expected,
            n_bins=bin_count,
            sample_weight=weights,
        ),
        ordinal_mae=ordinal_mean_absolute_error(
            true_labels,
            predicted_labels,
            classes=expected,
            sample_weight=weights,
        ),
        quadratic_weighted_kappa=float(
            cohen_kappa_score(
                true_labels,
                predicted_labels,
                labels=list(expected),
                weights="quadratic",
                sample_weight=weights,
            )
        ),
    )
