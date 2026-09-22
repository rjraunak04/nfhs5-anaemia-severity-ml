"""Fail CI if a public commit contains restricted data or common secret material."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESTRICTED_SUFFIXES = {
    ".parquet",
    ".pq",
    ".sav",
    ".dta",
    ".sas7bdat",
    ".joblib",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".sqlite3",
    ".db",
}
RESTRICTED_DIRECTORIES = {"runs", "artifacts", "checkpoints", "models", "mlruns"}
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".ipynb",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "OpenAI-style token": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
LOCAL_PATH_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
}
FORBIDDEN_DEMO_KEYS = {
    "caseid",
    "groups",
    "group_ids",
    "indices",
    "predictions",
    "probabilities",
    "psu",
    "psu_id",
    "raw_rows",
    "sample_weight",
    "train_indices",
    "validation_indices",
    "y_proba",
    "y_true",
}


def tracked_files() -> list[Path]:
    """Return Git-tracked files relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        Path(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def is_allowed_sample_csv(path: Path) -> bool:
    """Allow only explicitly synthetic sample CSV files."""
    return len(path.parts) >= 2 and path.parts[:2] == ("data", "sample")


def scan_path(path: Path) -> list[str]:
    """Return public-release violations for one tracked path."""
    problems: list[str] = []
    lowered_parts = {part.casefold() for part in path.parts}

    if lowered_parts & RESTRICTED_DIRECTORIES:
        problems.append("generated/restricted artifact directory is tracked")

    if path.suffix.casefold() in RESTRICTED_SUFFIXES:
        problems.append(f"restricted artifact type {path.suffix}")

    if path.suffix.casefold() == ".csv" and not is_allowed_sample_csv(path):
        problems.append("CSV outside data/sample is not allowed")

    if path.name == ".env":
        problems.append(".env must never be tracked")

    if path.suffix.casefold() not in TEXT_SUFFIXES and path.name != "Dockerfile":
        return problems

    full_path = ROOT / path
    try:
        text = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        problems.append("expected text file is not valid UTF-8")
        return problems

    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            problems.append(f"possible {name}")

    for name, pattern in LOCAL_PATH_PATTERNS.items():
        if pattern.search(text):
            problems.append(f"local machine path detected: {name}")

    if path.parts and path.parts[0] == "demo" and path.suffix.casefold() == ".json":
        for key in FORBIDDEN_DEMO_KEYS:
            if re.search(rf'"{re.escape(key)}"\s*:', text, flags=re.IGNORECASE):
                problems.append(f"row-level dashboard key is forbidden: {key}")

    return problems


def main() -> int:
    """Audit all tracked files and fail closed on unsafe public content."""
    violations: list[str] = []
    for path in tracked_files():
        for problem in scan_path(path):
            violations.append(f"{path.as_posix()}: {problem}")

    if violations:
        print("Public repository audit FAILED:")
        for violation in violations:
            print(f" - {violation}")
        return 1

    print("Public repository audit PASS: no blocked artifacts or common secret patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
