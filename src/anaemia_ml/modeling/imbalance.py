"""Training-fold-only class weighting and SMOTENC sensitivity planning."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from anaemia_ml.evaluation.config import validate_validation_config
from anaemia_ml.preprocessing import predictor_groups

DEFAULT_CLASSES = (0, 1, 2, 3)


class ImbalanceError(ValueError):
    """Raised when an imbalancelv operation is unsafe or inconsistent."""


@dataclass(frozen=True)
class ImbalanceSettings:
    """Validated imbalance settings from the confirmatory protocol."""

    primary_strategy: str
    smotenc_ratio: float
    random_seed: int


@dataclass(frozen=True)
class SMOTENCSpec:
    """Auditable parameters for a fold-local SMOTENC sensitivity run."""

    categorical_features: tuple[int, ...]
    sampling_strategy: dict[int, int]
    random_state: int
    k_neighbors: int

    def as_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by imblearn.SMOTENC."""
        return {
            "categorical_features": list(self.categorical_features),
            "sampling_strategy": dict(self.sampling_strategy),
            "random_state": self.random_state,
            "k_neighbors": self.k_neighbors,
        }


def _expected_classes(
    values: Sequence[int],
) -> tuple[int, ...]:
    raw_classes = tuple(values)

    if not raw_classes:
        raise ImbalanceError(
            "expected_classes must not be empty."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        for value in raw_classes
    ):
        raise ImbalanceError(
            "expected_classes must contain integers only."
        )

    classes = tuple(
        int(value)
        for value in raw_classes
    )

    if len(classes) != len(set(classes)):
        raise ImbalanceError(
            "expected_classes must not contain duplicates."
        )

    return classes


def _class_counts(
    y: Sequence[int] | np.ndarray,
    *,
    expected_classes: Sequence[int],
) -> dict[int, int]:
    classes = _expected_classes(
        expected_classes
    )
    labels = np.asarray(y)

    if labels.ndim != 1 or labels.size == 0:
        raise ImbalanceError(
            "y must be a non-empty one-dimensional sequence."
        )

    try:
        numeric = labels.astype(float)
    except (TypeError, ValueError) as error:
        raise ImbalanceError(
            "y must contain integer class labels."
        ) from error

    if (
        not np.isfinite(numeric).all()
        or not np.equal(
            numeric,
            np.floor(numeric),
        ).all()
    ):
        raise ImbalanceError(
            "y must contain finite integer class labels."
        )

    integer_labels = numeric.astype(int)
    observed = set(
        integer_labels.tolist()
    )

    unexpected = sorted(
        observed - set(classes)
    )
    missing = sorted(
        set(classes) - observed
    )

    if unexpected:
        raise ImbalanceError(
            f"Unexpected target classes: {unexpected}"
        )

    if missing:
        raise ImbalanceError(
            "Training fold is missing target "
            f"classes: {missing}"
        )

    counts = Counter(
        integer_labels.tolist()
    )

    return {
        label: counts[label]
        for label in classes
    }


def imbalance_settings(
    config: Mapping[str, Any],
) -> ImbalanceSettings:
    """Extract validated settings from validation.yaml data."""
    copied = dict(config)

    validate_validation_config(
        copied
    )

    preprocessing = copied[
        "preprocessing"
    ]

    return ImbalanceSettings(
        primary_strategy=preprocessing[
            "primary_imbalance_strategy"
        ],
        smotenc_ratio=float(
            preprocessing[
                "smotenc_sensitivity_ratio"
            ]
        ),
        random_seed=int(
            copied["random_seed"]
        ),
    )


def balanced_class_weights(
    y: Sequence[int] | np.ndarray,
    *,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
) -> dict[int, float]:
    """Compute weights using labels from one training fold only."""
    counts = _class_counts(
        y,
        expected_classes=expected_classes,
    )

    sample_count = sum(
        counts.values()
    )
    class_count = len(
        counts
    )

    return {
        label: sample_count / (
            class_count * count
        )
        for label, count in counts.items()
    }


def balanced_sample_weights(
    y: Sequence[int] | np.ndarray,
    *,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
) -> np.ndarray:
    """Return row weights with a training-fold mean of one."""
    labels = np.asarray(
        y
    ).astype(int)

    weights = balanced_class_weights(
        y,
        expected_classes=expected_classes,
    )

    return np.asarray(
        [
            weights[int(label)]
            for label in labels
        ],
        dtype=float,
    )


def smotenc_sampling_targets(
    y: Sequence[int] | np.ndarray,
    *,
    ratio: float,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
) -> dict[int, int]:
    """Plan multiclass oversampling relative to majority size."""
    if (
        isinstance(ratio, bool)
        or not isinstance(
            ratio,
            int | float,
        )
    ):
        raise ImbalanceError(
            "SMOTENC ratio must be numeric."
        )

    if not 0 < float(ratio) <= 1:
        raise ImbalanceError(
            "SMOTENC ratio must lie in (0, 1]."
        )

    counts = _class_counts(
        y,
        expected_classes=expected_classes,
    )

    target_size = math.ceil(
        float(ratio)
        * max(counts.values())
    )

    return {
        label: target_size
        for label, count in counts.items()
        if count < target_size
    }


def smotenc_categorical_indices(
    feature_schema: Mapping[str, Any],
    *,
    variant: str = "india_policy",
) -> tuple[int, ...]:
    """Return categorical positions in raw predictor order."""
    groups = predictor_groups(
        dict(feature_schema),
        variant=variant,
    )

    positions = {
        column: index
        for index, column in enumerate(
            groups.all
        )
    }

    return tuple(
        positions[column]
        for column in groups.categorical
    )


def make_smotenc_spec(
    y: Sequence[int] | np.ndarray,
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    variant: str = "india_policy",
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
    k_neighbors: int = 5,
) -> SMOTENCSpec:
    """Build a safe specification from one training fold."""
    if (
        isinstance(k_neighbors, bool)
        or not isinstance(
            k_neighbors,
            int,
        )
    ):
        raise ImbalanceError(
            "k_neighbors must be an integer."
        )

    if k_neighbors < 1:
        raise ImbalanceError(
            "k_neighbors must be at least 1."
        )

    settings = imbalance_settings(
        validation_config
    )

    counts = _class_counts(
        y,
        expected_classes=expected_classes,
    )

    targets = smotenc_sampling_targets(
        y,
        ratio=settings.smotenc_ratio,
        expected_classes=expected_classes,
    )

    insufficient = {
        label: counts[label]
        for label in targets
        if counts[label] <= k_neighbors
    }

    if insufficient:
        raise ImbalanceError(
            "SMOTENC training classes need more rows "
            f"than k_neighbors: {insufficient}"
        )

    return SMOTENCSpec(
        categorical_features=(
            smotenc_categorical_indices(
                feature_schema,
                variant=variant,
            )
        ),
        sampling_strategy=targets,
        random_state=settings.random_seed,
        k_neighbors=k_neighbors,
    )


def _validated_sample_weight(
    sample_weight: Sequence[float] | np.ndarray,
    *,
    sample_count: int,
) -> np.ndarray:
    values = np.asarray(
        sample_weight,
        dtype=float,
    )

    if values.shape != (
        sample_count,
    ):
        raise ImbalanceError(
            "sample_weight must contain one value "
            "per training row."
        )

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or values.sum() <= 0
    ):
        raise ImbalanceError(
            "sample_weight must be finite, "
            "non-negative, and non-zero."
        )

    return values


class FoldSafeClassWeightClassifier(
    ClassifierMixin,
    BaseEstimator,
):
    """Compute class weights independently during every fit."""

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        expected_classes: Sequence[int] = DEFAULT_CLASSES,
    ) -> None:
        self.estimator = estimator
        self.expected_classes = expected_classes

    def fit(
        self,
        X: Any,
        y: Sequence[int] | np.ndarray,
        sample_weight: (
            Sequence[float]
            | np.ndarray
            | None
        ) = None,
        **fit_params: Any,
    ) -> FoldSafeClassWeightClassifier:
        """Fit a cloned estimator using this training y only."""
        class_weights = balanced_class_weights(
            y,
            expected_classes=self.expected_classes,
        )

        estimator = clone(
            self.estimator
        )
        parameters = estimator.get_params(
            deep=False
        )

        if "class_weight" not in parameters:
            raise ImbalanceError(
                f"{type(estimator).__name__} does "
                "not expose class_weight."
            )

        existing = parameters[
            "class_weight"
        ]

        if (
            existing is not None
            and existing != "balanced"
        ):
            raise ImbalanceError(
                "The wrapped estimator must not "
                "contain custom class weights."
            )

        estimator.set_params(
            class_weight=class_weights
        )

        if sample_weight is not None:
            if not has_fit_parameter(
                estimator,
                "sample_weight",
            ):
                raise ImbalanceError(
                    f"{type(estimator).__name__} does "
                    "not accept sample_weight."
                )

            fit_params[
                "sample_weight"
            ] = _validated_sample_weight(
                sample_weight,
                sample_count=len(y),
            )

        estimator.fit(
            X,
            y,
            **fit_params,
        )

        self.estimator_ = estimator
        self.class_weight_ = class_weights
        self.class_count_ = _class_counts(
            y,
            expected_classes=self.expected_classes,
        )
        self.classes_ = np.asarray(
            estimator.classes_
        )

        if hasattr(
            estimator,
            "n_features_in_",
        ):
            self.n_features_in_ = (
                estimator.n_features_in_
            )

        if hasattr(
            estimator,
            "feature_names_in_",
        ):
            self.feature_names_in_ = (
                estimator.feature_names_in_
            )

        return self

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:
        """Predict using the fitted wrapped estimator."""
        check_is_fitted(
            self,
            "estimator_",
        )

        return self.estimator_.predict(
            X
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:
        """Return probabilities when supported."""
        check_is_fitted(
            self,
            "estimator_",
        )

        if not hasattr(
            self.estimator_,
            "predict_proba",
        ):
            raise AttributeError(
                "The wrapped estimator does not "
                "support predict_proba."
            )

        return self.estimator_.predict_proba(
            X
        )


def build_primary_imbalance_classifier(
    estimator: BaseEstimator,
    validation_config: Mapping[str, Any],
    *,
    expected_classes: Sequence[int] = DEFAULT_CLASSES,
) -> FoldSafeClassWeightClassifier:
    """Build the protocol-specified primary classifier."""
    settings = imbalance_settings(
        validation_config
    )

    if (
        settings.primary_strategy
        != "class_weight"
    ):
        raise ImbalanceError(
            "Only the prespecified class_weight "
            "strategy is allowed."
        )

    return FoldSafeClassWeightClassifier(
        estimator,
        expected_classes=expected_classes,
    )