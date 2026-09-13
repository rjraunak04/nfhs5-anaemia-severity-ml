"""Validated, versioned hyperparameter search-space configuration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from anaemia_ml.modeling.registry import model_spec, registered_models

PROTECTED_PARAMETERS = frozenset({"class_weight", "random_state", "n_jobs"})
SUPPORTED_PARAMETER_TYPES = frozenset({"float", "int", "categorical"})


class SearchSpaceError(ValueError):
    """Raised when a tuning search-space contract is invalid."""


class TrialProtocol(Protocol):
    """Subset of the Optuna Trial interface used by this project."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any: ...


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SearchSpaceError(f"{name} must be a mapping.")
    return value


def _exact_keys(
    mapping: Mapping[str, Any],
    *,
    expected: set[str],
    name: str,
) -> None:
    invalid_keys = [key for key in mapping if not isinstance(key, str)]
    if invalid_keys:
        raise SearchSpaceError(f"{name} keys must be strings.")
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise SearchSpaceError(f"{name} is missing keys: {missing}.")
    if unknown:
        raise SearchSpaceError(f"{name} contains unknown keys: {unknown}.")


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SearchSpaceError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise SearchSpaceError(f"{name} must be finite.")
    return number


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchSpaceError(f"{name} must be an integer.")
    return value


def _validate_float_parameter(specification: Mapping[str, Any], *, name: str) -> None:
    allowed = {"type", "low", "high", "step", "log"}
    unknown = sorted(specification.keys() - allowed)
    if unknown:
        raise SearchSpaceError(f"{name} contains unknown keys: {unknown}.")
    low = _finite_number(specification.get("low"), name=f"{name}.low")
    high = _finite_number(specification.get("high"), name=f"{name}.high")
    if low >= high:
        raise SearchSpaceError(f"{name}.low must be smaller than high.")
    log = specification.get("log", False)
    if not isinstance(log, bool):
        raise SearchSpaceError(f"{name}.log must be boolean.")
    if log and low <= 0:
        raise SearchSpaceError(f"{name}.low must be positive for logarithmic search.")
    step = specification.get("step")
    if step is not None:
        if _finite_number(step, name=f"{name}.step") <= 0:
            raise SearchSpaceError(f"{name}.step must be positive.")
        if log:
            raise SearchSpaceError(
                f"{name} cannot combine step with logarithmic search."
            )


def _validate_int_parameter(specification: Mapping[str, Any], *, name: str) -> None:
    allowed = {"type", "low", "high", "step", "log"}
    unknown = sorted(specification.keys() - allowed)
    if unknown:
        raise SearchSpaceError(f"{name} contains unknown keys: {unknown}.")
    low = _integer(specification.get("low"), name=f"{name}.low")
    high = _integer(specification.get("high"), name=f"{name}.high")
    if low >= high:
        raise SearchSpaceError(f"{name}.low must be smaller than high.")
    step = _integer(specification.get("step", 1), name=f"{name}.step")
    if step < 1:
        raise SearchSpaceError(f"{name}.step must be positive.")
    log = specification.get("log", False)
    if not isinstance(log, bool):
        raise SearchSpaceError(f"{name}.log must be boolean.")
    if log and low <= 0:
        raise SearchSpaceError(f"{name}.low must be positive for logarithmic search.")
    if log and step != 1:
        raise SearchSpaceError(f"{name} cannot combine step with logarithmic search.")


def _validate_categorical_parameter(
    specification: Mapping[str, Any],
    *,
    name: str,
) -> None:
    _exact_keys(specification, expected={"type", "choices"}, name=name)
    choices = specification["choices"]
    if (
        isinstance(choices, (str, bytes))
        or not isinstance(choices, list)
        or not choices
    ):
        raise SearchSpaceError(f"{name}.choices must be a non-empty list.")
    fingerprints = []
    for choice in choices:
        try:
            fingerprints.append(
                json.dumps(
                    choice, allow_nan=False, separators=(",", ":"), sort_keys=True
                )
            )
        except (TypeError, ValueError) as error:
            raise SearchSpaceError(
                f"{name}.choices must be JSON-safe and finite."
            ) from error
    if len(fingerprints) != len(set(fingerprints)):
        raise SearchSpaceError(f"{name}.choices must not contain duplicates.")


def _validate_parameter(specification: Any, *, name: str) -> None:
    parameter = _mapping(specification, name=name)
    parameter_type = parameter.get("type")
    if parameter_type not in SUPPORTED_PARAMETER_TYPES:
        raise SearchSpaceError(
            f"{name}.type must be one of {sorted(SUPPORTED_PARAMETER_TYPES)}."
        )
    if parameter_type == "float":
        _validate_float_parameter(parameter, name=name)
    elif parameter_type == "int":
        _validate_int_parameter(parameter, name=name)
    else:
        _validate_categorical_parameter(parameter, name=name)


def validate_search_space(config: Mapping[str, Any]) -> None:
    """Reject incomplete, unsafe, or unsupported search-space settings."""
    root = _mapping(config, name="search space")
    _exact_keys(
        root,
        expected={"search_space_version", "objective", "sampler", "models"},
        name="search space",
    )
    version = root["search_space_version"]
    if not isinstance(version, str) or not version.strip():
        raise SearchSpaceError("search_space_version must be a non-empty string.")

    objective = _mapping(root["objective"], name="objective")
    _exact_keys(objective, expected={"metric", "direction"}, name="objective")
    if objective["metric"] != "macro_f1" or objective["direction"] != "maximize":
        raise SearchSpaceError("The tuning objective must maximize macro_f1.")

    sampler = _mapping(root["sampler"], name="sampler")
    _exact_keys(sampler, expected={"name"}, name="sampler")
    if sampler["name"] != "TPESampler":
        raise SearchSpaceError("Only the seeded TPESampler is supported.")

    models = _mapping(root["models"], name="models")
    expected_models = {spec.name for spec in registered_models()}
    _exact_keys(models, expected=expected_models, name="models")
    for model_name, raw_parameters in models.items():
        parameters = _mapping(raw_parameters, name=f"models.{model_name}")
        if not parameters:
            raise SearchSpaceError(f"models.{model_name} must not be empty.")
        protected = sorted(PROTECTED_PARAMETERS & parameters.keys())
        if protected:
            raise SearchSpaceError(
                f"models.{model_name} contains protected parameters: {protected}."
            )
        for parameter_name, specification in parameters.items():
            if not isinstance(parameter_name, str) or not parameter_name:
                raise SearchSpaceError("Parameter names must be non-empty strings.")
            _validate_parameter(
                specification,
                name=f"models.{model_name}.{parameter_name}",
            )


def load_search_space(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML search-space contract."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Search-space config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise SearchSpaceError("Search-space YAML root must be a mapping.")
    validate_search_space(config)
    return config


def search_space_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for validated settings."""
    validate_search_space(config)
    canonical = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def suggest_parameters(
    trial: TrialProtocol,
    config: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Suggest one model parameter set through an Optuna-compatible trial."""
    validate_search_space(config)
    spec = model_spec(model_name)
    parameters = config["models"][spec.name]
    suggestions = {}
    for name, raw_specification in parameters.items():
        specification = dict(raw_specification)
        parameter_type = specification["type"]
        if parameter_type == "float":
            suggestions[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                step=(
                    None
                    if specification.get("step") is None
                    else float(specification["step"])
                ),
                log=bool(specification.get("log", False)),
            )
        elif parameter_type == "int":
            suggestions[name] = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
                log=bool(specification.get("log", False)),
            )
        else:
            suggestions[name] = trial.suggest_categorical(
                name,
                list(specification["choices"]),
            )
    return suggestions
