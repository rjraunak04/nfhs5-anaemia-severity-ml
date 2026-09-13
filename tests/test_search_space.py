"""Tests for the frozen hyperparameter search-space contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from anaemia_ml.modeling.search_space import (
    SearchSpaceError,
    load_search_space,
    search_space_fingerprint,
    suggest_parameters,
    validate_search_space,
)

ROOT = Path(__file__).parents[1]
SEARCH_SPACE_PATH = ROOT / "configs" / "search_space.yaml"


@pytest.fixture
def search_space() -> dict:
    return load_search_space(SEARCH_SPACE_PATH)


class FakeTrial:
    """Deterministic stand-in for the Optuna suggestion interface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        del high, step, log
        self.calls.append(("float", name))
        return low

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        del low, step, log
        self.calls.append(("int", name))
        return high

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.calls.append(("categorical", name))
        return choices[0]


def test_repository_search_space_is_valid_and_complete(search_space: dict) -> None:
    validate_search_space(search_space)

    assert tuple(search_space["models"]) == (
        "logistic_regression",
        "random_forest",
        "lightgbm",
    )
    assert search_space["objective"] == {
        "metric": "macro_f1",
        "direction": "maximize",
    }


def test_search_space_fingerprint_is_stable(search_space: dict) -> None:
    first = search_space_fingerprint(search_space)
    reordered = dict(reversed(list(search_space.items())))

    assert len(first) == 64
    assert first == search_space_fingerprint(reordered)


def test_float_parameter_routes_to_trial_interface(search_space: dict) -> None:
    trial = FakeTrial()

    parameters = suggest_parameters(trial, search_space, "logistic_regression")

    assert parameters == {"C": 0.01}
    assert trial.calls == [("float", "C")]


def test_tree_parameter_types_route_to_trial_interface(search_space: dict) -> None:
    trial = FakeTrial()

    parameters = suggest_parameters(trial, search_space, "random_forest")

    assert parameters == {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    }
    assert trial.calls == [
        ("int", "n_estimators"),
        ("categorical", "max_depth"),
        ("int", "min_samples_leaf"),
        ("categorical", "max_features"),
    ]


def test_protected_parameter_is_rejected(search_space: dict) -> None:
    invalid = copy.deepcopy(search_space)
    invalid["models"]["logistic_regression"]["random_state"] = {
        "type": "int",
        "low": 1,
        "high": 10,
    }

    with pytest.raises(SearchSpaceError, match="protected"):
        validate_search_space(invalid)


def test_logarithmic_float_cannot_have_step(search_space: dict) -> None:
    invalid = copy.deepcopy(search_space)
    invalid["models"]["logistic_regression"]["C"]["step"] = 0.1

    with pytest.raises(SearchSpaceError, match="cannot combine"):
        validate_search_space(invalid)


def test_all_registered_models_are_required(search_space: dict) -> None:
    invalid = copy.deepcopy(search_space)
    del invalid["models"]["lightgbm"]

    with pytest.raises(SearchSpaceError, match="missing keys"):
        validate_search_space(invalid)


def test_non_string_mapping_key_is_rejected_cleanly(search_space: dict) -> None:
    invalid = copy.deepcopy(search_space)
    invalid[1] = "unexpected"

    with pytest.raises(SearchSpaceError, match="keys must be strings"):
        validate_search_space(invalid)


def test_duplicate_categorical_choices_are_rejected(search_space: dict) -> None:
    invalid = copy.deepcopy(search_space)
    invalid["models"]["random_forest"]["max_depth"]["choices"] = [12, 12]

    with pytest.raises(SearchSpaceError, match="duplicates"):
        validate_search_space(invalid)
