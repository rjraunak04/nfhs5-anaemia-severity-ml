"""Training-fold-only preprocessing for prespecified model features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from anaemia_ml.features.schema import feature_columns

MISSING_CATEGORY = "__MISSING__"
ModelFamily = Literal["linear", "tree"]

NUMERIC_GROUPS = (
    "continuous_numeric",
    "count_numeric",
)

CATEGORICAL_GROUPS = (
    "nominal_categorical",
    "indicator_categorical",
    "ordinal_categorical",
)


class PreprocessingError(ValueError):
    """Raised when preprocessing settings or inputs are unsafe."""


@dataclass(frozen=True)
class PredictorGroups:
    """Ordered predictor columns used by the preprocessing pipeline."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    all: tuple[str, ...]


class PredictorFrameSelector(TransformerMixin, BaseEstimator):
    """Select only frozen predictors from a pandas DataFrame."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = columns

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "Predictor input must be a pandas DataFrame."
            )

        if frame.columns.has_duplicates:
            duplicates = frame.columns[
                frame.columns.duplicated()
            ].tolist()

            raise PreprocessingError(
                f"Duplicate input columns: {duplicates}"
            )

        missing = [
            column
            for column in self.columns
            if column not in frame.columns
        ]

        if missing:
            raise PreprocessingError(
                f"Missing frozen predictors: {missing}"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> PredictorFrameSelector:
        """Validate the training frame without learning statistics."""
        del y

        self._validate_frame(X)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return predictors in deterministic schema order."""
        check_is_fitted(self, "is_fitted_")
        self._validate_frame(X)

        return X.loc[:, list(self.columns)].copy()

    def get_feature_names_out(
    self,
    input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return selected predictor names."""
        del input_features

        check_is_fitted(self, "is_fitted_")

        return np.asarray(
            self.columns,
            dtype=object,
        )


def _as_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreprocessingError(
            f"{name} must be a mapping."
        )

    return value


def _group_columns(
    feature_groups: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[str, ...]:
    columns: list[str] = []

    for name in names:
        values = feature_groups.get(name)

        if (
            not isinstance(values, list)
            or any(
                not isinstance(column, str)
                or not column
                for column in values
            )
        ):
            raise PreprocessingError(
                f"feature_groups.{name} must contain "
                "non-empty strings."
            )

        columns.extend(values)

    return tuple(columns)


def _validate_preprocessing_contract(
    schema: Mapping[str, Any],
) -> None:
    contract = _as_mapping(
        schema.get("preprocessing_contract"),
        name="preprocessing_contract",
    )

    if contract.get("fit_scope") != "training_fold_only":
        raise PreprocessingError(
            "Preprocessing fit_scope must be "
            "training_fold_only."
        )

    numeric = _as_mapping(
        contract.get("numeric"),
        name="preprocessing_contract.numeric",
    )

    expected_numeric = {
        "imputation": "median",
        "add_missing_indicator": True,
        "scaling": "model_dependent",
    }

    for key, expected in expected_numeric.items():
        if numeric.get(key) != expected:
            raise PreprocessingError(
                "Unsupported numeric preprocessing rule: "
                f"{key}={numeric.get(key)!r}."
            )

    categorical = _as_mapping(
        contract.get("categorical"),
        name="preprocessing_contract.categorical",
    )

    expected_categorical = {
        "imputation": "explicit_missing_category",
        "unknown_category_policy": "handle_without_failure",
        "linear_model_encoding": "one_hot",
        "tree_model_encoding": "model_appropriate",
        "verify_ordinal_order_from_official_labels": True,
    }

    for key, expected in expected_categorical.items():
        if categorical.get(key) != expected:
            raise PreprocessingError(
                "Unsupported categorical preprocessing rule: "
                f"{key}={categorical.get(key)!r}."
            )

    engineered = _as_mapping(
        contract.get("engineered_features"),
        name="preprocessing_contract.engineered_features",
    )

    if engineered.get("primary_analysis") != []:
        raise PreprocessingError(
            "Primary engineered features must remain empty."
        )

    if (
        engineered.get("post_hoc_features_allowed")
        is not False
    ):
        raise PreprocessingError(
            "Post-hoc engineered features must remain disabled."
        )

    guards = _as_mapping(
        schema.get("leakage_guards"),
        name="leakage_guards",
    )

    if (
        guards.get("preprocessing_before_split_allowed")
        is not False
    ):
        raise PreprocessingError(
            "The schema must forbid preprocessing "
            "before splitting."
        )


def predictor_groups(
    schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
) -> PredictorGroups:
    """Resolve predictor groups for one frozen model variant."""
    _validate_preprocessing_contract(schema)

    groups = _as_mapping(
        schema.get("feature_groups"),
        name="feature_groups",
    )

    selected = tuple(
        feature_columns(
            dict(schema),
            variant=variant,
        )
    )
    selected_set = set(selected)

    numeric = tuple(
        column
        for column in _group_columns(
            groups,
            NUMERIC_GROUPS,
        )
        if column in selected_set
    )

    categorical = tuple(
        column
        for column in _group_columns(
            groups,
            CATEGORICAL_GROUPS,
        )
        if column in selected_set
    )

    combined = numeric + categorical

    if (
        len(combined) != len(set(combined))
        or set(combined) != selected_set
    ):
        raise PreprocessingError(
            "Numeric and categorical groups do not "
            "partition the selected predictors."
        )

    guards = _as_mapping(
        schema.get("leakage_guards"),
        name="leakage_guards",
    )

    forbidden = set(
        guards.get(
            "forbidden_predictors",
            [],
        )
    )

    overlap = sorted(
        selected_set & forbidden
    )

    if overlap:
        raise PreprocessingError(
            "Selected predictors include forbidden "
            f"columns: {overlap}"
        )

    return PredictorGroups(
        numeric=numeric,
        categorical=categorical,
        all=selected,
    )


def _categorical_strings(
    values: Any,
) -> Any:
    """Convert coded categories and represent missingness explicitly."""
    is_frame = isinstance(
        values,
        pd.DataFrame,
    )

    frame = (
        values.copy()
        if is_frame
        else pd.DataFrame(values)
    )

    result = (
        frame.astype("string")
        .fillna(MISSING_CATEGORY)
        .astype(object)
    )

    if is_frame:
        return result

    return result.to_numpy(
        dtype=object,
    )


def build_preprocessor(
    schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    model_family: ModelFamily = "linear",
    sparse_output: bool = True,
) -> Pipeline:
    """Build an unfitted transformer for use inside model CV."""
    if model_family not in {
        "linear",
        "tree",
    }:
        raise PreprocessingError(
            f"Unknown model_family {model_family!r}; "
            "expected 'linear' or 'tree'."
        )

    groups = predictor_groups(
        schema,
        variant=variant,
    )

    numeric_steps: list[
        tuple[str, Any]
    ] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]

    if model_family == "linear":
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    # All categorical variables are one-hot encoded until
    # verified ordinal category orders are separately frozen.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "missing_category",
                FunctionTransformer(
                    _categorical_strings,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=numeric_steps,
                ),
                list(groups.numeric),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(groups.categorical),
            ),
        ],
        remainder="drop",
        sparse_threshold=(
            1.0
            if sparse_output
            else 0.0
        ),
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            (
                "select_predictors",
                PredictorFrameSelector(
                    groups.all,
                ),
            ),
            (
                "columns",
                column_transformer,
            ),
        ]
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    model_family: ModelFamily = "linear",
    sparse_output: bool = True,
) -> Pipeline:
    """Keep preprocessing and estimation together during CV."""
    if estimator is None:
        raise TypeError(
            "estimator must not be None."
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    schema,
                    variant=variant,
                    model_family=model_family,
                    sparse_output=sparse_output,
                ),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def transformed_feature_names(
    preprocessor: Pipeline,
) -> tuple[str, ...]:
    """Return output feature names after fitting."""
    check_is_fitted(
        preprocessor
    )

    names = (
        preprocessor
        .named_steps["columns"]
        .get_feature_names_out()
    )

    return tuple(
        str(name)
        for name in names
    )