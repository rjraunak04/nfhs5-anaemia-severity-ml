"""Tests for validation configuration and final-test safeguards."""

from copy import deepcopy
from pathlib import Path

import pytest

from anaemia_ml.evaluation.config import (
    FINAL_TEST_UNLOCK_TOKEN,
    FinalTestLockedError,
    ValidationConfigError,
    load_validation_config,
    require_final_test_unlock,
    validate_validation_config,
)

CONFIG_PATH = Path("configs/validation.yaml")


@pytest.fixture
def validation_config():
    """Return the repository validation configuration."""
    return load_validation_config(CONFIG_PATH)


def test_repository_validation_config_loads(validation_config):
    assert validation_config["validation_version"] == "1.0.0"
    assert validation_config["nested_cv"]["outer_folds"] == 5
    assert validation_config["nested_cv"]["inner_folds"] == 3
    assert len(validation_config["final_test"]["required_gates"]) == 7


def test_partition_fractions_must_sum_to_one(validation_config):
    invalid_config = deepcopy(validation_config)
    invalid_config["partitions"]["locked_test"] = 0.30

    with pytest.raises(ValidationConfigError, match="sum to 1.0"):
        validate_validation_config(invalid_config)


def test_nested_cv_is_limited_to_development_data(validation_config):
    invalid_config = deepcopy(validation_config)
    invalid_config["nested_cv"]["scope"] = "all_data"

    with pytest.raises(ValidationConfigError, match="development_only"):
        validate_validation_config(invalid_config)


def test_required_gates_cannot_be_missing(validation_config):
    invalid_config = deepcopy(validation_config)
    del invalid_config["final_test"]["required_gates"]

    with pytest.raises(ValidationConfigError, match="required_gates"):
        validate_validation_config(invalid_config)


def test_raw_rows_cannot_be_allowed(validation_config):
    invalid_config = deepcopy(validation_config)
    invalid_config["privacy"]["allow_raw_rows_in_repository"] = True

    with pytest.raises(ValidationConfigError, match="Unsafe"):
        validate_validation_config(invalid_config)


def test_final_test_rejects_incomplete_gates(validation_config):
    gates = {
        gate: True
        for gate in validation_config["final_test"]["required_gates"]
    }
    gates["calibration_frozen"] = False

    with pytest.raises(FinalTestLockedError, match="Incomplete"):
        require_final_test_unlock(
            validation_config,
            gates,
            environ={},
        )


def test_final_test_rejects_wrong_unlock_token(validation_config):
    gates = {
        gate: True
        for gate in validation_config["final_test"]["required_gates"]
    }
    variable = validation_config["final_test"]["unlock_environment_variable"]

    with pytest.raises(FinalTestLockedError, match="Final test is locked"):
        require_final_test_unlock(
            validation_config,
            gates,
            environ={variable: "wrong-token"},
        )


def test_final_test_unlocks_only_after_all_safeguards(validation_config):
    gates = {
        gate: True
        for gate in validation_config["final_test"]["required_gates"]
    }
    variable = validation_config["final_test"]["unlock_environment_variable"]

    require_final_test_unlock(
        validation_config,
        gates,
        environ={variable: FINAL_TEST_UNLOCK_TOKEN},
    )