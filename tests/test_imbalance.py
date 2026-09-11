"""Tests for fold-local imbalance handling."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from anaemia_ml.evaluation.config import load_validation_config
from anaemia_ml.features.schema import load_feature_schema
from anaemia_ml.modeling import (
    FoldSafeClassWeightClassifier,
    ImbalanceError,
    balanced_class_weights,
    balanced_sample_weights,
    build_primary_imbalance_classifier,
    imbalance_settings,
    make_smotenc_spec,
    smotenc_categorical_indices,
    smotenc_sampling_targets,
)

ROOT = Path(__file__).parents[1]
VALIDATION_PATH = ROOT / "configs" / "validation.yaml"
FEATURE_SCHEMA_PATH = ROOT / "configs" / "feature_schema.yaml"


@pytest.fixture
def validation_config() -> dict:
    """Load the repository validation configuration."""
    return load_validation_config(
        VALIDATION_PATH
    )


@pytest.fixture
def feature_schema() -> dict:
    """Load the repository feature schema."""
    return load_feature_schema(
        FEATURE_SCHEMA_PATH
    )


def labels(
    *counts: int,
) -> np.ndarray:
    """Create synthetic ordered-class labels."""
    return np.concatenate(
        [
            np.repeat(
                label,
                count,
            )
            for label, count
            in enumerate(counts)
        ]
    )


def features(
    sample_count: int,
) -> np.ndarray:
    """Create synthetic numeric predictors."""
    values = np.arange(
        sample_count * 3,
        dtype=float,
    )

    return values.reshape(
        sample_count,
        3,
    )


def test_settings_are_read_from_validated_config(
    validation_config: dict,
) -> None:
    settings = imbalance_settings(
        validation_config
    )

    assert (
        settings.primary_strategy
        == "class_weight"
    )
    assert settings.smotenc_ratio == pytest.approx(
        0.45
    )
    assert settings.random_seed == 42


def test_balanced_class_weight_formula() -> None:
    weights = balanced_class_weights(
        labels(
            8,
            4,
            2,
            2,
        )
    )

    assert weights == pytest.approx(
        {
            0: 0.5,
            1: 1.0,
            2: 2.0,
            3: 2.0,
        }
    )


def test_balanced_sample_weights_have_mean_one() -> None:
    weights = balanced_sample_weights(
        labels(
            8,
            4,
            2,
            2,
        )
    )

    assert weights.mean() == pytest.approx(
        1.0
    )
    assert weights.shape == (
        16,
    )


def test_missing_training_class_is_rejected() -> None:
    with pytest.raises(
        ImbalanceError,
        match="missing target classes",
    ):
        balanced_class_weights(
            labels(
                8,
                4,
                2,
                0,
            )
        )


def test_unexpected_training_class_is_rejected() -> None:
    target = np.append(
        labels(
            8,
            4,
            2,
            2,
        ),
        9,
    )

    with pytest.raises(
        ImbalanceError,
        match="Unexpected target classes",
    ):
        balanced_class_weights(
            target
        )


def test_multiclass_smotenc_targets_do_not_undersample() -> None:
    targets = smotenc_sampling_targets(
        labels(
            20,
            12,
            8,
            6,
        ),
        ratio=0.45,
    )

    assert targets == {
        2: 9,
        3: 9,
    }


def test_smotenc_indices_follow_variant_order(
    feature_schema: dict,
) -> None:
    india = smotenc_categorical_indices(
        feature_schema
    )

    transportable = smotenc_categorical_indices(
        feature_schema,
        variant="transportable",
    )

    assert india == tuple(
        range(
            6,
            32,
        )
    )

    assert transportable == tuple(
        range(
            6,
            31,
        )
    )


def test_smotenc_spec_uses_protocol_settings(
    feature_schema: dict,
    validation_config: dict,
) -> None:
    spec = make_smotenc_spec(
        labels(
            20,
            12,
            8,
            6,
        ),
        feature_schema,
        validation_config,
    )

    assert spec.sampling_strategy == {
        2: 9,
        3: 9,
    }
    assert spec.random_state == 42
    assert spec.k_neighbors == 5

    assert spec.as_kwargs()[
        "categorical_features"
    ] == list(
        range(
            6,
            32,
        )
    )


def test_smotenc_rejects_insufficient_neighbors(
    feature_schema: dict,
    validation_config: dict,
) -> None:
    with pytest.raises(
        ImbalanceError,
        match="more rows than k_neighbors",
    ):
        make_smotenc_spec(
            labels(
                20,
                12,
                8,
                5,
            ),
            feature_schema,
            validation_config,
        )


def test_classifier_learns_weights_from_fit_labels() -> None:
    target = labels(
        8,
        4,
        2,
        2,
    )

    model = FoldSafeClassWeightClassifier(
        LogisticRegression(
            max_iter=500
        )
    )

    model.fit(
        features(len(target)),
        target,
    )

    assert model.class_weight_ == pytest.approx(
        {
            0: 0.5,
            1: 1.0,
            2: 2.0,
            3: 2.0,
        }
    )

    predictions = model.predict(
        features(len(target))
    )

    probabilities = model.predict_proba(
        features(len(target))
    )

    assert predictions.shape == (
        16,
    )
    assert probabilities.shape == (
        16,
        4,
    )


def test_separate_fit_calls_produce_separate_weights() -> None:
    first_target = labels(
        8,
        4,
        2,
        2,
    )

    second_target = labels(
        4,
        4,
        4,
        4,
    )

    first = FoldSafeClassWeightClassifier(
        LogisticRegression(
            max_iter=500
        )
    )

    second = FoldSafeClassWeightClassifier(
        LogisticRegression(
            max_iter=500
        )
    )

    first.fit(
        features(len(first_target)),
        first_target,
    )

    second.fit(
        features(len(second_target)),
        second_target,
    )

    assert (
        first.class_weight_
        != second.class_weight_
    )

    assert second.class_weight_ == {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
    }


def test_estimator_without_class_weight_is_rejected() -> None:
    target = labels(
        4,
        4,
        4,
        4,
    )

    model = FoldSafeClassWeightClassifier(
        KNeighborsClassifier()
    )

    with pytest.raises(
        ImbalanceError,
        match="expose class_weight",
    ):
        model.fit(
            features(len(target)),
            target,
        )


def test_invalid_sample_weight_is_rejected() -> None:
    target = labels(
        4,
        4,
        4,
        4,
    )

    model = FoldSafeClassWeightClassifier(
        LogisticRegression(
            max_iter=500
        )
    )

    invalid_weights = np.ones(
        len(target)
    )
    invalid_weights[0] = -1

    with pytest.raises(
        ImbalanceError,
        match="non-negative",
    ):
        model.fit(
            features(len(target)),
            target,
            sample_weight=invalid_weights,
        )


def test_primary_builder_returns_fold_safe_classifier(
    validation_config: dict,
) -> None:
    model = build_primary_imbalance_classifier(
        LogisticRegression(
            max_iter=500
        ),
        validation_config,
    )

    assert isinstance(
        model,
        FoldSafeClassWeightClassifier,
    )