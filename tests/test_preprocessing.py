"""Tests for training-fold-only preprocessing."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, cross_validate

from anaemia_ml.features.schema import load_feature_schema
from anaemia_ml.preprocessing import (
    MISSING_CATEGORY,
    PreprocessingError,
    build_model_pipeline,
    build_preprocessor,
    predictor_groups,
    transformed_feature_names,
)

ROOT = Path(__file__).parents[1]
SCHEMA_PATH = (
    ROOT
    / "configs"
    / "feature_schema.yaml"
)


@pytest.fixture
def schema() -> dict:
    """Load the validated frozen feature schema."""
    return load_feature_schema(
        SCHEMA_PATH
    )


def synthetic_frame(
    schema: dict,
) -> pd.DataFrame:
    """Create synthetic data without restricted NFHS records."""
    feature_groups = schema[
        "feature_groups"
    ]

    numeric_columns = (
        feature_groups[
            "continuous_numeric"
        ]
        + feature_groups[
            "count_numeric"
        ]
    )

    categorical_columns = (
        feature_groups[
            "nominal_categorical"
        ]
        + feature_groups[
            "indicator_categorical"
        ]
        + feature_groups[
            "ordinal_categorical"
        ]
    )

    data: dict[
        str,
        pd.Series | np.ndarray,
    ] = {}

    for column in numeric_columns:
        data[column] = np.asarray(
            [
                1.0,
                2.0,
                100.0,
                101.0,
            ],
            dtype=float,
        )

    for column in categorical_columns:
        data[column] = pd.Series(
            [
                1.0,
                2.0,
                1.0,
                2.0,
            ],
            dtype="Float64",
        )

    frame = pd.DataFrame(
        data
    )

    # Extra non-predictors must never reach the transformer.
    frame["v003"] = [
        10,
        11,
        12,
        13,
    ]
    frame["v457"] = [
        4,
        3,
        2,
        1,
    ]

    return frame


def test_predictor_groups_respect_model_variants(
    schema: dict,
) -> None:
    india = predictor_groups(
        schema,
        variant="india_policy",
    )

    transportable = predictor_groups(
        schema,
        variant="transportable",
    )

    assert len(india.all) == 32
    assert len(transportable.all) == 31
    assert "v024" in india.categorical
    assert "v024" not in transportable.all


def test_builder_returns_unfitted_preprocessor(
    schema: dict,
) -> None:
    transformer = build_preprocessor(
        schema
    )

    with pytest.raises(
        NotFittedError,
    ):
        transformer.transform(
            synthetic_frame(schema)
        )


def test_non_predictors_are_not_transformed(
    schema: dict,
) -> None:
    transformer = build_preprocessor(
        schema,
        sparse_output=False,
    )

    transformer.fit(
        synthetic_frame(schema)
    )

    names = transformed_feature_names(
        transformer
    )

    assert not any(
        "v003" in name
        or "v457" in name
        for name in names
    )


def test_unseen_categories_and_missing_values_are_supported(
    schema: dict,
) -> None:
    train = synthetic_frame(
        schema
    )

    train.loc[
        1,
        "v130",
    ] = pd.NA

    validation = train.iloc[
        [3]
    ].copy()

    validation.loc[
        validation.index[0],
        "v130",
    ] = 999.0

    validation.loc[
        validation.index[0],
        "v024",
    ] = pd.NA

    transformer = build_preprocessor(
        schema,
        sparse_output=False,
    )

    train_output = transformer.fit_transform(
        train
    )

    validation_output = transformer.transform(
        validation
    )

    assert (
        validation_output.shape[1]
        == train_output.shape[1]
    )

    assert np.isfinite(
        validation_output
    ).all()

    assert any(
        MISSING_CATEGORY in name
        for name in transformed_feature_names(
            transformer
        )
    )


def test_scaling_is_model_dependent(
    schema: dict,
) -> None:
    linear = build_preprocessor(
        schema,
        model_family="linear",
    )

    tree = build_preprocessor(
        schema,
        model_family="tree",
    )

    linear_numeric = (
        linear
        .named_steps["columns"]
        .transformers[0][1]
    )

    tree_numeric = (
        tree
        .named_steps["columns"]
        .transformers[0][1]
    )

    assert (
        "scaler"
        in linear_numeric.named_steps
    )

    assert (
        "scaler"
        not in tree_numeric.named_steps
    )


def test_each_cv_fit_learns_training_fold_median(
    schema: dict,
) -> None:
    frame = synthetic_frame(
        schema
    )

    target = np.asarray(
        [
            0,
            1,
            0,
            1,
        ]
    )

    groups = np.asarray(
        [
            0,
            0,
            1,
            1,
        ]
    )

    splitter = GroupKFold(
        n_splits=2
    )

    splits = list(
        splitter.split(
            frame,
            target,
            groups,
        )
    )

    result = cross_validate(
        build_model_pipeline(
            DummyClassifier(
                strategy="most_frequent"
            ),
            schema,
            model_family="tree",
            sparse_output=False,
        ),
        frame,
        target,
        groups=groups,
        cv=splits,
        return_estimator=True,
    )

    learned_medians: list[
        float
    ] = []

    expected_medians: list[
        float
    ] = []

    for estimator, (
        train_indices,
        _,
    ) in zip(
        result["estimator"],
        splits,
        strict=True,
    ):
        imputer = (
            estimator
            .named_steps[
                "preprocessor"
            ]
            .named_steps[
                "columns"
            ]
            .named_transformers_[
                "numeric"
            ]
            .named_steps[
                "imputer"
            ]
        )

        learned_medians.append(
            float(
                imputer.statistics_[0]
            )
        )

        expected_medians.append(
            float(
                frame.iloc[
                    train_indices
                ]["v012"].median()
            )
        )

    assert learned_medians == pytest.approx(
        expected_medians
    )

    assert set(
        learned_medians
    ) == {
        1.5,
        100.5,
    }


def test_missing_required_predictor_is_rejected(
    schema: dict,
) -> None:
    incomplete = synthetic_frame(
        schema
    ).drop(
        columns="v012"
    )

    with pytest.raises(
        PreprocessingError,
        match="Missing frozen predictors",
    ):
        build_preprocessor(
            schema
        ).fit(
            incomplete
        )


def test_duplicate_input_column_is_rejected(
    schema: dict,
) -> None:
    frame = synthetic_frame(
        schema
    )

    duplicate = pd.concat(
        [
            frame,
            frame[["v012"]],
        ],
        axis=1,
    )

    with pytest.raises(
        PreprocessingError,
        match="Duplicate input columns",
    ):
        build_preprocessor(
            schema
        ).fit(
            duplicate
        )


def test_unsafe_fit_scope_is_rejected(
    schema: dict,
) -> None:
    invalid = deepcopy(
        schema
    )

    invalid[
        "preprocessing_contract"
    ][
        "fit_scope"
    ] = "full_dataset"

    with pytest.raises(
        PreprocessingError,
        match="training_fold_only",
    ):
        build_preprocessor(
            invalid
        )


def test_unknown_model_family_is_rejected(
    schema: dict,
) -> None:
    with pytest.raises(
        PreprocessingError,
        match="Unknown model_family",
    ):
        build_preprocessor(
            schema,
            model_family="neural",
        )