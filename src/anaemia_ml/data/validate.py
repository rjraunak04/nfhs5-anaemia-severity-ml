"""Validate the restricted NFHS-5 research extract against its data contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

IssueLevel = Literal["error", "warning"]

PREDICTOR_ROLE_KEYS = (
    "primary_geographic_predictor",
    "sociodemographic_predictors",
    "reproductive_predictors",
    "anthropometry_access_environment_predictors",
    "nutrition_comorbidity_predictors",
)


class DataContractError(ValueError):
    """Raised when a contract or dataset violates a required rule."""


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding."""

    level: IssueLevel
    code: str
    message: str


@dataclass
class ValidationReport:
    """Structured validation result suitable for CI and audit records."""

    issues: list[ValidationIssue] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True when the report contains no errors."""
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return error-level findings."""
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return warning-level findings."""
        return [issue for issue in self.issues if issue.level == "warning"]

    def add(self, level: IssueLevel, code: str, message: str) -> None:
        """Append a finding."""
        self.issues.append(ValidationIssue(level=level, code=code, message=message))

    def raise_for_errors(self) -> None:
        """Raise a compact exception if validation failed."""
        if self.errors:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors)
            raise DataContractError(details)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "statistics": self.statistics,
            "issues": [asdict(issue) for issue in self.issues],
        }


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a YAML data contract."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(f"Data contract not found: {contract_path}")

    with contract_path.open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)

    if not isinstance(contract, dict):
        raise DataContractError("The YAML root must be a mapping.")

    validate_contract_definition(contract)
    return contract


def validate_contract_definition(contract: dict[str, Any]) -> None:
    """Reject internally inconsistent contract definitions."""
    required_sections = {
        "dataset",
        "expected_column_order",
        "column_roles",
        "never_use_as_predictors",
        "target",
        "survey_design",
        "respondent_key",
        "basic_validation",
    }
    missing_sections = sorted(required_sections - contract.keys())
    if missing_sections:
        raise DataContractError(f"Missing contract sections: {missing_sections}")

    expected = contract["expected_column_order"]
    if not isinstance(expected, list) or not expected:
        raise DataContractError("expected_column_order must be a non-empty list.")
    if len(expected) != len(set(expected)):
        raise DataContractError("expected_column_order contains duplicate names.")

    declared_count = contract["dataset"].get("expected_columns")
    if declared_count != len(expected):
        raise DataContractError(
            f"expected_columns={declared_count} but {len(expected)} names are declared."
        )

    expected_set = set(expected)
    role_columns = {
        column
        for columns in contract["column_roles"].values()
        for column in columns
    }
    unknown_role_columns = sorted(role_columns - expected_set)
    if unknown_role_columns:
        raise DataContractError(f"Role columns missing from schema: {unknown_role_columns}")

    never_predictors = set(contract["never_use_as_predictors"])
    unknown_exclusions = sorted(never_predictors - expected_set)
    if unknown_exclusions:
        raise DataContractError(f"Unknown leakage exclusions: {unknown_exclusions}")

    declared_predictors = {
        column
        for role in PREDICTOR_ROLE_KEYS
        for column in contract["column_roles"].get(role, [])
    }
    overlap = sorted(declared_predictors & never_predictors)
    if overlap:
        raise DataContractError(f"Predictors overlap leakage exclusions: {overlap}")

    target = contract["target"]
    if target["raw_column"] not in expected_set:
        raise DataContractError("The primary target is absent from expected columns.")
    mapped_classes = sorted(set(target["raw_to_model_mapping"].values()))
    if mapped_classes != target["ordered_classes"]:
        raise DataContractError(
            "Target mapping values do not match target.ordered_classes."
        )


def predictor_columns(
    contract: dict[str, Any], *, transportable: bool = False
) -> list[str]:
    """Return prespecified predictors in stable contract order."""
    roles = contract["column_roles"]
    selected = {
        column
        for role in PREDICTOR_ROLE_KEYS
        for column in roles.get(role, [])
    }
    if transportable:
        selected -= set(roles.get("primary_geographic_predictor", []))
    excluded = set(contract["never_use_as_predictors"])
    return [
        column
        for column in contract["expected_column_order"]
        if column in selected and column not in excluded
    ]


def _numeric(series: pd.Series, report: ValidationReport, column: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid_count = int((series.notna() & converted.isna()).sum())
    if invalid_count:
        report.add(
            "error",
            "non_numeric_values",
            f"{column} contains {invalid_count} non-numeric values.",
        )
    return converted


def _validate_allowed_values(
    frame: pd.DataFrame,
    report: ValidationReport,
    rule_name: str,
    rule: dict[str, Any],
) -> None:
    column = rule["column"]
    values = _numeric(frame[column], report, column)
    allowed = set(rule["allowed_raw_values"])
    invalid = sorted(values.dropna().loc[~values.dropna().isin(allowed)].unique().tolist())
    if invalid:
        report.add(
            "error",
            "invalid_allowed_values",
            f"{rule_name}/{column} contains disallowed values: {invalid[:10]}.",
        )


def _profile_cardinalities(
    frame: pd.DataFrame, contract: dict[str, Any], report: ValidationReport
) -> None:
    survey = contract["survey_design"]
    observed = {
        "states_and_union_territories": int(frame[survey["state_column"]].nunique(dropna=True)),
        "districts": int(frame[survey["district_column"]].nunique(dropna=True)),
        "composite_psus": int(
            frame[survey["composite_psu_components"]].drop_duplicates().shape[0]
        ),
        "composite_strata": int(
            frame[survey["composite_stratum_components"]].drop_duplicates().shape[0]
        ),
    }
    report.statistics.update(observed)

    reference = contract.get("reference_profile", {})
    for key, observed_value in observed.items():
        expected_value = reference.get(key)
        if expected_value is not None and observed_value != expected_value:
            report.add(
                "warning",
                "reference_profile_mismatch",
                f"{key}: observed {observed_value}, reference {expected_value}.",
            )


def validate_dataframe(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    strict_profile: bool = True,
) -> ValidationReport:
    """Validate a dataframe without mutating it."""
    validate_contract_definition(contract)
    report = ValidationReport(
        statistics={"rows": int(len(frame)), "columns": int(frame.shape[1])}
    )

    expected = contract["expected_column_order"]
    expected_set = set(expected)
    actual = frame.columns.tolist()
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    if missing:
        report.add("error", "missing_columns", f"Missing columns: {missing}")
    if extra and contract["dataset"].get("strict_column_set", False):
        report.add("error", "unexpected_columns", f"Unexpected columns: {extra}")
    if not missing and not extra and actual != expected:
        report.add("error", "column_order_mismatch", "Columns are not in contract order.")
    if missing:
        return report

    if strict_profile:
        expected_rows = int(contract["dataset"]["expected_rows"])
        if len(frame) != expected_rows:
            report.add(
                "error",
                "row_count_mismatch",
                f"Observed {len(frame)} rows; expected {expected_rows}.",
            )

    required_non_null = {
        contract["target"]["raw_column"],
        contract["survey_design"]["weight_column"],
        contract["survey_design"]["psu_column"],
        contract["survey_design"]["stratum_column"],
        contract["survey_design"]["state_column"],
    }
    for column in sorted(required_non_null):
        missing_count = int(frame[column].isna().sum())
        if missing_count:
            report.add(
                "error",
                "missing_required_values",
                f"{column} contains {missing_count} missing values.",
            )

    rules = contract["basic_validation"]
    age_rule = rules["age"]
    age = _numeric(frame[age_rule["column"]], report, age_rule["column"])
    invalid_age = age.notna() & ~age.between(age_rule["minimum"], age_rule["maximum"])
    if invalid_age.any():
        report.add(
            "error",
            "invalid_age",
            f"{int(invalid_age.sum())} ages fall outside "
            f"[{age_rule['minimum']}, {age_rule['maximum']}].",
        )

    _validate_allowed_values(frame, report, "residence", rules["residence"])
    _validate_allowed_values(frame, report, "wealth_index", rules["wealth_index"])
    _validate_allowed_values(frame, report, "target", rules["target"])

    weight_column = rules["survey_weight"]["column"]
    weights = _numeric(frame[weight_column], report, weight_column)
    if rules["survey_weight"].get("must_be_positive", False):
        invalid_weight = weights.notna() & (weights <= 0)
        if invalid_weight.any():
            report.add(
                "error",
                "nonpositive_weight",
                f"{int(invalid_weight.sum())} survey weights are non-positive.",
            )

    key_rule = contract["respondent_key"]
    duplicate_count = int(frame.duplicated(subset=key_rule["columns"], keep=False).sum())
    report.statistics["duplicate_respondent_rows"] = duplicate_count
    if duplicate_count and not key_rule.get("duplicates_allowed", False):
        report.add(
            "error",
            "duplicate_respondent_key",
            f"{duplicate_count} rows share a respondent key.",
        )

    target_column = contract["target"]["raw_column"]
    target_numeric = pd.to_numeric(frame[target_column], errors="coerce")
    report.statistics["target_counts_raw"] = {
        str(int(key) if float(key).is_integer() else key): int(value)
        for key, value in target_numeric.value_counts(dropna=False).sort_index().items()
        if pd.notna(key)
    }

    _profile_cardinalities(frame, contract, report)
    report.statistics["primary_predictor_count"] = len(predictor_columns(contract))
    report.statistics["transportable_predictor_count"] = len(
        predictor_columns(contract, transportable=True)
    )
    return report


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a supported restricted dataset format."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)
    raise DataContractError(f"Unsupported dataset format: {suffix}")


def validate_file(
    data_path: str | Path,
    contract_path: str | Path,
    *,
    strict_profile: bool = True,
) -> ValidationReport:
    """Read a dataset and validate it against a contract."""
    contract = load_contract(contract_path)
    frame = read_dataset(data_path)
    return validate_dataframe(frame, contract, strict_profile=strict_profile)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Restricted CSV or Parquet dataset.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/data_contract.yaml"),
        help="YAML data contract.",
    )
    parser.add_argument(
        "--no-strict-profile",
        action="store_true",
        help="Skip exact full-dataset row-count enforcement.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON audit-report path.")
    return parser


def main() -> int:
    """Run the command-line validator."""
    args = _build_parser().parse_args()
    report = validate_file(
        args.data_path,
        args.contract,
        strict_profile=not args.no_strict_profile,
    )
    payload = report.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
