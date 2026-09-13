"""Group-safe nested CV over reproducible parameter candidates.

The final research run may generate candidates with Optuna. This module owns
the leakage boundary: candidate selection uses inner folds only, while outer
folds are used once for unbiased development performance estimation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from anaemia_ml.evaluation.config import validate_validation_config
from anaemia_ml.evaluation.grouped_validation import nested_group_splits
from anaemia_ml.evaluation.metrics import MulticlassMetrics
from anaemia_ml.modeling.registry import build_estimator, model_spec
from anaemia_ml.modeling.runner import fit_evaluate_model

SEARCH_METHOD = "bounded_parameter_candidates"


class NestedCVError(ValueError):
    """Raised when nested model selection would be unsafe or ambiguous."""


@dataclass(frozen=True)
class CandidateScore:
    """Inner-fold macro-F1 values for one parameter candidate."""

    candidate_number: int
    parameters: dict[str, Any]
    inner_macro_f1: tuple[float, ...]
    mean_macro_f1: float
    standard_deviation: float

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe candidate summary."""
        return {
            "candidate_number": self.candidate_number,
            "parameters": dict(self.parameters),
            "inner_macro_f1": list(self.inner_macro_f1),
            "mean_macro_f1": self.mean_macro_f1,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True)
class MetricAggregate:
    """Across-outer-fold summary for one evaluation metric."""

    name: str
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float]:
        """Return JSON-safe aggregate values."""
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class OuterFoldResult:
    """Selection audit and untouched outer-fold metrics."""

    fold_number: int
    selected_candidate_number: int
    selected_parameters: dict[str, Any]
    candidate_scores: tuple[CandidateScore, ...]
    training_rows: int
    validation_rows: int
    metrics: MulticlassMetrics

    def summary(self) -> dict[str, Any]:
        """Return aggregate-safe output without rows, groups, or indices."""
        return {
            "fold_number": self.fold_number,
            "selected_candidate_number": self.selected_candidate_number,
            "selected_parameters": dict(self.selected_parameters),
            "candidate_scores": [score.summary() for score in self.candidate_scores],
            "training_rows": self.training_rows,
            "validation_rows": self.validation_rows,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class NestedCVReport:
    """Auditable development-only nested cross-validation result."""

    model_name: str
    display_name: str
    variant: str
    primary_metric: str
    search_method: str
    random_seed: int
    outer_fold_results: tuple[OuterFoldResult, ...]
    aggregate_metrics: tuple[MetricAggregate, ...]
    total_model_fits: int

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe report without restricted row-level values."""
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "variant": self.variant,
            "scope": "development_only",
            "primary_metric": self.primary_metric,
            "search_method": self.search_method,
            "random_seed": self.random_seed,
            "outer_folds": [fold.summary() for fold in self.outer_fold_results],
            "aggregate_metrics": {
                aggregate.name: aggregate.as_dict()
                for aggregate in self.aggregate_metrics
            },
            "total_model_fits": self.total_model_fits,
        }


_DEFAULT_PARAMETER_CANDIDATES = {
    "logistic_regression": (
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ),
    "random_forest": (
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 4},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 8},
    ),
    "lightgbm": (
        {"num_leaves": 15, "min_child_samples": 20},
        {"num_leaves": 31, "min_child_samples": 30},
        {"num_leaves": 63, "min_child_samples": 50},
    ),
}


def default_parameter_candidates(model_name: str) -> tuple[dict[str, Any], ...]:
    """Return fresh copies of deterministic smoke-search candidates."""
    spec = model_spec(model_name)
    return tuple(
        dict(parameters) for parameters in _DEFAULT_PARAMETER_CANDIDATES[spec.name]
    )


def _validated_vector(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size != expected_length:
        raise NestedCVError(
            f"{name} must be one-dimensional with {expected_length} values."
        )
    if bool(pd.isna(vector).any()):
        raise NestedCVError(f"{name} must not contain missing values.")
    return vector


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
        raise NestedCVError("sample_weight must contain numeric values.") from error
    if weights.shape != (sample_count,):
        raise NestedCVError("sample_weight must contain one value per development row.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise NestedCVError("Development sample weights must be finite and positive.")
    return weights


def _validated_candidates(
    model_name: str,
    candidates: Sequence[Mapping[str, Any]] | None,
    *,
    random_seed: int,
    n_jobs: int,
    search_budget: int,
) -> tuple[dict[str, Any], ...]:
    raw_candidates = (
        default_parameter_candidates(model_name) if candidates is None else candidates
    )
    if isinstance(raw_candidates, (str, bytes, Mapping)):
        raise NestedCVError("parameter_candidates must be a sequence of mappings.")
    copied_candidates = tuple(raw_candidates)
    if not copied_candidates:
        raise NestedCVError("parameter_candidates must not be empty.")
    if len(copied_candidates) > search_budget:
        raise NestedCVError(
            f"Parameter candidate count {len(copied_candidates)} exceeds the "
            f"configured search budget of {search_budget}."
        )

    validated = []
    fingerprints = set()
    for candidate in copied_candidates:
        if not isinstance(candidate, Mapping):
            raise NestedCVError("Every parameter candidate must be a mapping.")
        parameters = dict(candidate)
        try:
            fingerprint = json.dumps(
                parameters,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise NestedCVError(
                "Parameter candidates must contain JSON-safe finite values."
            ) from error
        if fingerprint in fingerprints:
            raise NestedCVError("parameter_candidates must not contain duplicates.")
        fingerprints.add(fingerprint)
        build_estimator(
            model_name,
            random_seed=random_seed,
            n_jobs=n_jobs,
            parameters=parameters,
        )
        validated.append(parameters)
    return tuple(validated)


def _sample_standard_deviation(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _candidate_score(
    candidate_number: int,
    parameters: Mapping[str, Any],
    inner_macro_f1: Sequence[float],
) -> CandidateScore:
    scores = tuple(float(value) for value in inner_macro_f1)
    return CandidateScore(
        candidate_number=candidate_number,
        parameters=dict(parameters),
        inner_macro_f1=scores,
        mean_macro_f1=float(np.mean(scores)),
        standard_deviation=_sample_standard_deviation(scores),
    )


def _select_candidate(scores: Sequence[CandidateScore]) -> CandidateScore:
    return max(
        scores,
        key=lambda score: (score.mean_macro_f1, -score.candidate_number),
    )


def _aggregate_metrics(
    outer_results: Sequence[OuterFoldResult],
) -> tuple[MetricAggregate, ...]:
    metric_rows = [fold.metrics.as_dict() for fold in outer_results]
    aggregates = []
    for name in metric_rows[0]:
        values = np.asarray([row[name] for row in metric_rows], dtype=float)
        aggregates.append(
            MetricAggregate(
                name=name,
                mean=float(values.mean()),
                standard_deviation=float(values.std(ddof=1)),
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
        )
    return tuple(aggregates)


def run_grouped_nested_cv(
    model_name: str,
    X_development: pd.DataFrame,
    y_development: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    feature_schema: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    *,
    parameter_candidates: Sequence[Mapping[str, Any]] | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    variant: str = "india_policy",
    n_jobs: int = -1,
) -> NestedCVReport:
    """Select parameters on inner folds and score untouched outer folds."""
    if not isinstance(X_development, pd.DataFrame) or X_development.empty:
        raise NestedCVError("X_development must be a non-empty pandas DataFrame.")
    if not isinstance(validation_config, Mapping):
        raise NestedCVError("validation_config must be a mapping.")
    copied_config = dict(validation_config)
    validate_validation_config(copied_config)

    sample_count = len(X_development)
    target = _validated_vector(
        y_development,
        name="y_development",
        expected_length=sample_count,
    )
    group_values = _validated_vector(
        groups,
        name="groups",
        expected_length=sample_count,
    )
    weights = _validated_sample_weight(sample_weight, sample_count=sample_count)

    nested = copied_config["nested_cv"]
    selection = copied_config["model_selection"]
    random_seed = int(copied_config["random_seed"])
    outer_count = int(nested["outer_folds"])
    inner_count = int(nested["inner_folds"])
    search_budget = int(selection["optuna_trials_per_outer_fold"])
    candidates = _validated_candidates(
        model_name,
        parameter_candidates,
        random_seed=random_seed,
        n_jobs=n_jobs,
        search_budget=search_budget,
    )
    folds = nested_group_splits(
        target,
        group_values,
        outer_splits=outer_count,
        inner_splits=inner_count,
        random_state=random_seed,
    )

    outer_results = []
    for fold in folds:
        candidate_scores = []
        for candidate_number, parameters in enumerate(candidates, start=1):
            inner_scores = []
            for inner_fold in fold.inner:
                train_indices = inner_fold.train_indices
                validation_indices = inner_fold.validation_indices
                run = fit_evaluate_model(
                    model_name,
                    X_development.iloc[train_indices],
                    target[train_indices],
                    group_values[train_indices],
                    X_development.iloc[validation_indices],
                    target[validation_indices],
                    group_values[validation_indices],
                    feature_schema,
                    copied_config,
                    variant=variant,
                    train_sample_weight=(
                        None if weights is None else weights[train_indices]
                    ),
                    validation_sample_weight=(
                        None if weights is None else weights[validation_indices]
                    ),
                    n_jobs=n_jobs,
                    parameters=parameters,
                )
                inner_scores.append(run.metrics.macro_f1)
            candidate_scores.append(
                _candidate_score(candidate_number, parameters, inner_scores)
            )

        selected = _select_candidate(candidate_scores)
        outer_train = fold.outer.train_indices
        outer_validation = fold.outer.validation_indices
        outer_run = fit_evaluate_model(
            model_name,
            X_development.iloc[outer_train],
            target[outer_train],
            group_values[outer_train],
            X_development.iloc[outer_validation],
            target[outer_validation],
            group_values[outer_validation],
            feature_schema,
            copied_config,
            variant=variant,
            train_sample_weight=None if weights is None else weights[outer_train],
            validation_sample_weight=(
                None if weights is None else weights[outer_validation]
            ),
            n_jobs=n_jobs,
            parameters=selected.parameters,
        )
        outer_results.append(
            OuterFoldResult(
                fold_number=fold.number,
                selected_candidate_number=selected.candidate_number,
                selected_parameters=dict(selected.parameters),
                candidate_scores=tuple(candidate_scores),
                training_rows=outer_run.training_rows,
                validation_rows=outer_run.validation_rows,
                metrics=outer_run.metrics,
            )
        )

    spec = model_spec(model_name)
    total_fits = outer_count * (len(candidates) * inner_count + 1)
    return NestedCVReport(
        model_name=spec.name,
        display_name=spec.display_name,
        variant=variant,
        primary_metric=str(selection["primary_metric"]),
        search_method=SEARCH_METHOD,
        random_seed=random_seed,
        outer_fold_results=tuple(outer_results),
        aggregate_metrics=_aggregate_metrics(outer_results),
        total_model_fits=total_fits,
    )
