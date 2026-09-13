"""Tests for protocol-defined multiclass evaluation metrics."""

import json

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from anaemia_ml.evaluation import (
    MetricError,
    align_probability_columns,
    evaluate_multiclass,
    expected_calibration_error,
    multiclass_brier_score,
    ordinal_mean_absolute_error,
)


@pytest.fixture
def evaluation_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic four-class evaluation example."""
    y_true = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    y_proba = np.asarray(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.45, 0.35, 0.15, 0.05],
            [0.10, 0.70, 0.15, 0.05],
            [0.15, 0.35, 0.40, 0.10],
            [0.05, 0.15, 0.70, 0.10],
            [0.05, 0.10, 0.35, 0.50],
            [0.02, 0.03, 0.15, 0.80],
            [0.05, 0.10, 0.55, 0.30],
        ]
    )
    y_pred = np.argmax(y_proba, axis=1)
    return y_true, y_pred, y_proba


def test_perfect_predictions_produce_expected_limits() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    y_proba = np.eye(4)

    report = evaluate_multiclass(y_true, y_true, y_proba)

    assert report.macro_f1 == pytest.approx(1.0)
    assert report.weighted_f1 == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx(1.0)
    assert report.macro_roc_auc_ovr == pytest.approx(1.0)
    assert report.severe_recall == pytest.approx(1.0)
    assert report.severe_precision == pytest.approx(1.0)
    assert report.severe_pr_auc == pytest.approx(1.0)
    assert report.multiclass_brier == pytest.approx(0.0)
    assert report.log_loss == pytest.approx(0.0, abs=1e-12)
    assert report.expected_calibration_error == pytest.approx(0.0)
    assert report.ordinal_mae == pytest.approx(0.0)
    assert report.quadratic_weighted_kappa == pytest.approx(1.0)


def test_report_matches_sklearn_reference_metrics(
    evaluation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = evaluation_data
    report = evaluate_multiclass(y_true, y_pred, y_proba)

    assert report.macro_f1 == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert report.macro_roc_auc_ovr == pytest.approx(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    )
    assert report.severe_pr_auc == pytest.approx(
        average_precision_score(y_true == 3, y_proba[:, 3])
    )
    assert report.quadratic_weighted_kappa == pytest.approx(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )


def test_multiclass_brier_uses_unscaled_definition() -> None:
    y_true = np.asarray([0, 1])
    y_proba = np.asarray(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.10, 0.60, 0.20, 0.10],
        ]
    )
    one_hot = np.eye(4)[y_true]
    expected = np.square(y_proba - one_hot).sum(axis=1).mean()

    assert multiclass_brier_score(y_true, y_proba) == pytest.approx(expected)


def test_top_label_ece_uses_equal_width_bins() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    y_proba = np.asarray(
        [
            [0.55, 0.15, 0.15, 0.15],
            [0.65, 0.35, 0.00, 0.00],
            [0.10, 0.10, 0.75, 0.05],
            [0.02, 0.01, 0.02, 0.95],
        ]
    )

    assert expected_calibration_error(
        y_true,
        y_proba,
        n_bins=5,
    ) == pytest.approx(0.225)


def test_ordinal_mae_follows_declared_class_order() -> None:
    y_true = np.asarray([10, 20, 30, 40])
    y_pred = np.asarray([20, 20, 10, 40])

    score = ordinal_mean_absolute_error(
        y_true,
        y_pred,
        classes=(10, 20, 30, 40),
    )

    assert score == pytest.approx(0.75)


def test_sample_weights_are_applied_consistently(
    evaluation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = evaluation_data
    weights = np.asarray([1, 1, 1, 1, 1, 1, 1, 20], dtype=float)

    unweighted = evaluate_multiclass(y_true, y_pred, y_proba)
    weighted = evaluate_multiclass(
        y_true,
        y_pred,
        y_proba,
        sample_weight=weights,
    )

    assert weighted.macro_f1 != pytest.approx(unweighted.macro_f1)
    assert weighted.ordinal_mae > unweighted.ordinal_mae


def test_report_dictionary_is_json_serializable(
    evaluation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred, y_proba = evaluation_data
    payload = evaluate_multiclass(y_true, y_pred, y_proba).as_dict()

    assert set(payload) == {
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "macro_roc_auc_ovr",
        "severe_recall",
        "severe_precision",
        "severe_pr_auc",
        "multiclass_brier",
        "log_loss",
        "expected_calibration_error",
        "ordinal_mae",
        "quadratic_weighted_kappa",
    }
    json.dumps(payload, allow_nan=False)


def test_probability_columns_can_be_aligned() -> None:
    canonical = np.asarray(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.40, 0.30, 0.20, 0.10],
        ]
    )
    model_order = (3, 1, 0, 2)
    shuffled = canonical[:, [3, 1, 0, 2]]

    aligned = align_probability_columns(
        shuffled,
        model_order,
    )

    assert aligned == pytest.approx(canonical)


def test_alignment_rejects_missing_model_class() -> None:
    with pytest.raises(MetricError, match="match expected_classes"):
        align_probability_columns(
            np.asarray([[0.20, 0.30, 0.50]]),
            (0, 1, 2),
        )


@pytest.mark.parametrize(
    ("y_proba", "message"),
    [
        (np.full((4, 3), 1 / 3), "shape"),
        (np.full((4, 4), 0.20), "sum to 1"),
        (
            np.asarray(
                [
                    [np.nan, 0.30, 0.30, 0.40],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ),
            "finite",
        ),
    ],
)
def test_invalid_probability_matrices_are_rejected(
    y_proba: np.ndarray,
    message: str,
) -> None:
    y_true = np.asarray([0, 1, 2, 3])

    with pytest.raises(MetricError, match=message):
        evaluate_multiclass(y_true, y_true, y_proba)


def test_missing_true_class_is_rejected() -> None:
    y_true = np.asarray([0, 0, 1, 2])
    y_proba = np.eye(4)

    with pytest.raises(MetricError, match="missing expected classes"):
        evaluate_multiclass(y_true, y_true, y_proba)


def test_unexpected_predicted_class_is_rejected() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    y_pred = np.asarray([0, 1, 2, 9])

    with pytest.raises(MetricError, match="Unexpected y_pred"):
        evaluate_multiclass(y_true, y_pred, np.eye(4))


def test_invalid_sample_weight_is_rejected() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    weights = np.asarray([1.0, 1.0, -1.0, 1.0])

    with pytest.raises(MetricError, match="non-negative"):
        evaluate_multiclass(
            y_true,
            y_true,
            np.eye(4),
            sample_weight=weights,
        )


def test_zero_weight_class_is_rejected() -> None:
    y_true = np.asarray([0, 1, 2, 3])
    weights = np.asarray([1.0, 1.0, 1.0, 0.0])

    with pytest.raises(MetricError, match="positive total sample weight"):
        evaluate_multiclass(
            y_true,
            y_true,
            np.eye(4),
            sample_weight=weights,
        )


def test_prediction_length_must_match_truth() -> None:
    with pytest.raises(MetricError, match="same length"):
        evaluate_multiclass(
            np.asarray([0, 1, 2, 3]),
            np.asarray([0, 1, 2]),
            np.eye(4),
        )


def test_invalid_severe_class_is_rejected() -> None:
    y_true = np.asarray([0, 1, 2, 3])

    with pytest.raises(MetricError, match="severe_class"):
        evaluate_multiclass(
            y_true,
            y_true,
            np.eye(4),
            severe_class=9,
        )


def test_invalid_ece_bin_count_is_rejected() -> None:
    y_true = np.asarray([0, 1, 2, 3])

    with pytest.raises(MetricError, match="n_bins"):
        expected_calibration_error(
            y_true,
            np.eye(4),
            n_bins=1,
        )
