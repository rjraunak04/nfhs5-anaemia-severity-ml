"""Tests for group-safe multiclass temperature scaling."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from anaemia_ml.evaluation.calibration import (
    CalibrationError,
    TemperatureScaledClassifier,
    apply_temperature,
    select_temperature_scaling,
)


def calibration_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.tile(np.arange(4), 8)
    groups = np.repeat(np.arange(8), 4)
    scores = np.full((target.size, 4), 0.4 / 3.0)
    scores[np.arange(target.size), target] = 0.6
    return target, scores, groups


def test_temperature_scaling_preserves_rows_and_class_order() -> None:
    _, scores, _ = calibration_case()

    scaled = apply_temperature(scores, 0.5)

    assert scaled.shape == scores.shape
    np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(scaled, axis=1), np.argmax(scores, axis=1))


def test_group_cross_fit_accepts_improving_temperature() -> None:
    target, scores, groups = calibration_case()

    result = select_temperature_scaling(
        target,
        scores,
        groups,
        sample_weight=np.linspace(0.7, 1.3, target.size),
        n_splits=4,
    )

    assert result.accepted is True
    assert result.method == "temperature_scaling"
    assert result.temperature < 1.0
    assert result.cross_fit_splits == 4
    assert result.cross_fitted_candidate.log_loss < result.before.log_loss
    assert result.as_dict()["locked_test_evaluated"] is False


def test_identity_is_available_as_safe_fallback() -> None:
    target, _, groups = calibration_case()
    scores = np.full((target.size, 4), 0.25)

    result = select_temperature_scaling(target, scores, groups, n_splits=4)

    assert result.accepted is False
    assert result.method == "identity"
    assert result.temperature == 1.0


def test_calibration_rejects_insufficient_class_groups() -> None:
    target = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    scores = np.full((8, 4), 0.25)

    with pytest.raises(CalibrationError, match="two distinct PSU groups"):
        select_temperature_scaling(target, scores, groups)


def test_temperature_wrapper_scales_a_fitted_estimator() -> None:
    features = np.arange(96, dtype=float).reshape(32, 3)
    target, _, _ = calibration_case()
    estimator = LogisticRegression(max_iter=500).fit(features, target)
    wrapper = TemperatureScaledClassifier(estimator, temperature=1.5)

    scores = wrapper.predict_proba(features)

    assert scores.shape == (32, 4)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)
    assert wrapper.predict(features).shape == (32,)
    with pytest.raises(CalibrationError, match="cannot be refit"):
        wrapper.fit(features, target)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), True])
def test_invalid_temperature_is_rejected(temperature) -> None:
    with pytest.raises(CalibrationError, match="temperature"):
        apply_temperature(np.array([[0.5, 0.5]]), temperature)
