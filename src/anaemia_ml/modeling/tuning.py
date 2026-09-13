"""Persistent, resumable Optuna tuning over the frozen search space."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from anaemia_ml.modeling.registry import ModelRegistryError, model_spec
from anaemia_ml.modeling.search_space import (
    search_space_fingerprint,
    suggest_parameters,
    validate_search_space,
)


class OptunaDependencyError(ImportError):
    """Raised when tuning is requested without the modeling dependencies."""


class TuningError(ValueError):
    """Raised when an Optuna study cannot be safely created or resumed."""


@dataclass(frozen=True)
class TuningResult:
    """JSON-safe result from one persistent bounded Optuna study."""

    study_name: str
    model_name: str
    experiment_fingerprint: str
    search_space_fingerprint: str
    random_seed: int
    trial_budget: int
    total_trials: int
    completed_trials: int
    best_trial_number: int
    best_value: float
    best_parameters: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return reproducibility metadata and the best completed trial."""
        return {
            "study_name": self.study_name,
            "model_name": self.model_name,
            "experiment_fingerprint": self.experiment_fingerprint,
            "search_space_fingerprint": self.search_space_fingerprint,
            "random_seed": self.random_seed,
            "trial_budget": self.trial_budget,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "best_trial_number": self.best_trial_number,
            "best_value": self.best_value,
            "best_parameters": dict(self.best_parameters),
        }


def _positive_integer(value: int, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise TuningError(f"{name} must be a {qualifier} integer.")
    return int(value)


def _non_empty_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TuningError(f"{name} must be a non-empty string.")
    return value


def _optuna_module() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise OptunaDependencyError(
            "Optuna tuning requires the modeling dependencies. Install them with "
            '`python -m pip install -e ".[modeling]"`.'
        ) from error
    return optuna


def _storage_url(path: Path) -> str:
    resolved = path.resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _validated_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TuningError("The tuning evaluator must return a numeric macro-F1 score.")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise TuningError("The tuning evaluator must return finite macro-F1 in [0, 1].")
    return score


def tune_parameters(
    model_name: str,
    evaluate_parameters: Callable[[Mapping[str, Any]], float],
    search_space: Mapping[str, Any],
    *,
    trial_budget: int,
    random_seed: int,
    storage_path: str | Path,
    study_name: str,
    experiment_fingerprint: str,
) -> TuningResult:
    """Run or resume one seeded, single-worker Optuna study.

    The SQLite database persists all attempted trials. The total number of
    stored trials, including failed trials, counts against ``trial_budget``.
    """
    if not callable(evaluate_parameters):
        raise TuningError("evaluate_parameters must be callable.")
    if not isinstance(search_space, Mapping):
        raise TuningError("search_space must be a mapping.")
    validate_search_space(search_space)
    try:
        checked_model_name = model_spec(model_name).name
    except ModelRegistryError as error:
        raise TuningError(str(error)) from error
    budget = _positive_integer(trial_budget, name="trial_budget")
    seed = _positive_integer(random_seed, name="random_seed", allow_zero=True)
    checked_study_name = _non_empty_text(study_name, name="study_name")
    checked_fingerprint = _non_empty_text(
        experiment_fingerprint,
        name="experiment_fingerprint",
    )
    database_path = Path(storage_path)
    if database_path.exists() and not database_path.is_file():
        raise TuningError(f"storage_path is not a file: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    optuna = _optuna_module()
    space_fingerprint = search_space_fingerprint(search_space)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=checked_study_name,
        storage=_storage_url(database_path),
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    expected_attributes = {
        "model_name": checked_model_name,
        "experiment_fingerprint": checked_fingerprint,
        "search_space_fingerprint": space_fingerprint,
        "random_seed": seed,
    }
    for name, expected in expected_attributes.items():
        stored = study.user_attrs.get(name)
        if stored is not None and stored != expected:
            raise TuningError(
                f"Stored study attribute {name!r} does not match this experiment."
            )
    if study.trials and any(
        name not in study.user_attrs for name in expected_attributes
    ):
        raise TuningError(
            "Stored study is missing required experiment identity metadata."
        )
    for name, value in expected_attributes.items():
        study.set_user_attr(name, value)

    existing_trials = len(study.trials)
    if existing_trials > budget:
        raise TuningError(
            f"Stored study already has {existing_trials} trials, exceeding budget {budget}."
        )

    def objective(trial: Any) -> float:
        parameters = suggest_parameters(trial, search_space, checked_model_name)
        return _validated_score(evaluate_parameters(parameters))

    remaining_trials = budget - existing_trials
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise TuningError("The study contains no successfully completed trials.")
    best = study.best_trial
    return TuningResult(
        study_name=checked_study_name,
        model_name=checked_model_name,
        experiment_fingerprint=checked_fingerprint,
        search_space_fingerprint=space_fingerprint,
        random_seed=seed,
        trial_budget=budget,
        total_trials=len(study.trials),
        completed_trials=len(completed),
        best_trial_number=int(best.number),
        best_value=float(best.value),
        best_parameters=dict(best.params),
    )
