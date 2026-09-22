# Contributing

Thank you for contributing to the NFHS-5 Anaemia Severity ML project.

## Development environment

This project supports Python 3.12.

Create a virtual environment with `python -m venv .venv`.

Install the project and development dependencies with:

`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`

## Branch workflow

Never develop directly on `main`.

1. Switch to `main`.
2. Pull the latest changes with `git pull --ff-only origin main`.
3. Create a focused feature branch.
4. Make one logical change.
5. Run all local quality checks.
6. Push the branch and open a pull request.
7. Merge only after GitHub Actions passes.
8. Use squash merging to keep the main history clean.

Recommended branch prefixes:

- `feat/` for new functionality
- `fix/` for bug fixes
- `data/` for data-contract or processing changes
- `test/` for tests
- `docs/` for documentation
- `ci/` for automation
- `chore/` for maintenance

## Local quality checks

Before opening a pull request, run:

- `.\.venv\Scripts\python.exe scripts/public_repo_audit.py`
- `.\.venv\Scripts\python.exe -m ruff check src tests scripts app.py`
- `.\.venv\Scripts\python.exe -m pytest --cov=anaemia_ml --cov-report=term-missing`
- `docker build -t nfhs5-anaemia-severity-ml .`

## Notebook requirements

Committed notebooks must:

- contain no saved outputs or execution counts;
- contain no credentials, API keys or tokens;
- use reproducible configuration and deterministic seeds;
- store checkpoints and generated artifacts outside Git;
- pass the notebook integrity check in GitHub Actions.

## Data governance

NFHS and DHS microdata are restricted research data.

Never commit:

- raw or reduced respondent-level datasets;
- CSV, SAV, DTA or Parquet research extracts;
- respondent identifiers or row-level predictions;
- model checkpoints or serialized fitted models;
- API keys, tokens or environment secrets.

Only documentation and explicitly approved synthetic sample data may be committed.

## Scientific safeguards

Every change must preserve:

- the prespecified 40-column data contract;
- the correct ordered anaemia target mapping;
- target-leakage exclusions;
- survey-weight, strata and PSU-aware analysis;
- PSU-disjoint development, calibration and test splits;
- fold-local preprocessing;
- locked final-test evaluation;
- checkpoint compatibility through configuration fingerprints.

Changes to eligibility, outcome definitions, predictors, splitting, calibration or evaluation must be documented in the pull request.

## Commit messages

Use concise Conventional Commit-style messages, for example:

- `feat(data): add schema validation`
- `fix(model): prevent preprocessing leakage`
- `test(data): cover invalid target codes`
- `docs: explain restricted-data policy`
- `ci: add automated quality gates`

## Pull requests

Each pull request should:

- contain one focused logical change;
- explain why the change is needed;
- list validation commands and results;
- contain no restricted data or generated artifacts;
- pass all GitHub Actions checks;
- document scientific and reproducibility implications.