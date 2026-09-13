"""Deterministic estimator registry for protocol-approved model families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ModelFamily = Literal["linear", "tree"]


class ModelRegistryError(ValueError):
    """Raised when a model request violates the frozen registry contract."""


class ModelDependencyError(ImportError):
    """Raised when an optional registered model dependency is unavailable."""


@dataclass(frozen=True)
class ModelSpec:
    """Stable metadata used to build one model pipeline."""

    name: str
    display_name: str
    family: ModelFamily
    sparse_output: bool
    optional_dependency: str | None = None

    @property
    def is_optional(self) -> bool:
        """Return whether the estimator needs a modeling extra."""
        return self.optional_dependency is not None


_MODEL_SPECS = (
    ModelSpec(
        name="logistic_regression",
        display_name="Multinomial Logistic Regression",
        family="linear",
        sparse_output=True,
    ),
    ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        family="tree",
        sparse_output=True,
    ),
    ModelSpec(
        name="lightgbm",
        display_name="LightGBM",
        family="tree",
        sparse_output=True,
        optional_dependency="lightgbm",
    ),
)

_MODEL_INDEX = {spec.name: spec for spec in _MODEL_SPECS}
_PROTECTED_PARAMETERS = frozenset({"class_weight", "random_state", "n_jobs"})


def registered_models() -> tuple[ModelSpec, ...]:
    """Return protocol models in deterministic comparison order."""
    return _MODEL_SPECS


def model_spec(name: str) -> ModelSpec:
    """Resolve a registered model name to immutable metadata."""
    if not isinstance(name, str) or not name.strip():
        raise ModelRegistryError("Model name must be a non-empty string.")

    try:
        return _MODEL_INDEX[name]
    except KeyError as error:
        available = ", ".join(_MODEL_INDEX)
        raise ModelRegistryError(
            f"Unknown model {name!r}; registered models: {available}."
        ) from error


def _validated_integer(
    value: int,
    *,
    name: str,
    allow_negative_one: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelRegistryError(f"{name} must be an integer.")

    integer = int(value)
    if allow_negative_one:
        if integer == 0 or integer < -1:
            raise ModelRegistryError(f"{name} must be -1 or a positive integer.")
    elif integer < 0:
        raise ModelRegistryError(f"{name} must be a non-negative integer.")

    return integer


def _validated_parameters(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ModelRegistryError("parameters must be a mapping.")

    copied = dict(parameters)
    invalid_keys = [key for key in copied if not isinstance(key, str) or not key]
    if invalid_keys:
        raise ModelRegistryError("Model parameter names must be non-empty strings.")

    protected = sorted(_PROTECTED_PARAMETERS & copied.keys())
    if protected:
        raise ModelRegistryError(
            "Reproducibility and imbalance parameters cannot be overridden: "
            f"{protected}."
        )

    return copied


def _base_estimator(
    name: str,
    *,
    random_seed: int,
    n_jobs: int,
) -> BaseEstimator:
    if name == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2_000,
            random_state=random_seed,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=n_jobs,
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ModelDependencyError(
                "LightGBM is optional. Install the modeling dependencies with "
                '`python -m pip install -e ".[modeling]"`.'
            ) from error

        return LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample_freq=1,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    raise AssertionError(f"Registry factory missing for {name!r}.")


def build_estimator(
    name: str,
    *,
    random_seed: int = 42,
    n_jobs: int = -1,
    parameters: Mapping[str, Any] | None = None,
) -> BaseEstimator:
    """Construct an unfitted estimator with reproducibility controls."""
    spec = model_spec(name)
    seed = _validated_integer(random_seed, name="random_seed")
    jobs = _validated_integer(
        n_jobs,
        name="n_jobs",
        allow_negative_one=True,
    )
    overrides = _validated_parameters(parameters)
    estimator = _base_estimator(
        spec.name,
        random_seed=seed,
        n_jobs=jobs,
    )

    if overrides:
        try:
            estimator.set_params(**overrides)
        except ValueError as error:
            raise ModelRegistryError(
                f"Invalid parameters for {name!r}: {sorted(overrides)}."
            ) from error

    return estimator
