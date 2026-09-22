# Public engineering release status

**Status:** Complete for the public software/portfolio release. Confirmatory research evaluation remains protocol-gated.

## What is complete

The repository now contains a reproducible Python package, a strict 40-column data contract, leakage-safe feature handling, PSU-disjoint development/calibration/test partitions, grouped nested cross-validation, calibrated model selection, aggregate-only SHAP reporting, integrity-checked model artifacts, automated tests, CI, a recruiter-facing Streamlit dashboard, software citation metadata, Docker packaging, deployment configuration and an automated public-repository safety audit.

The published NFHS evidence is intentionally limited to development and calibration aggregates. The dashboard contains no respondent rows, real PSU identifiers, row-level predictions, probabilities, fitted model binaries or restricted NFHS/DHS files.

## What “complete” does not mean

This public release does not claim that the scientific validation protocol is finished. The locked final-test partition has not been evaluated, so no final model-performance, clinical screening, subgroup-fairness, geographic-robustness or external-validation claim is made.

Those are confirmatory research steps, not missing software-engineering tasks. They require the authorized restricted dataset and must be executed only under the frozen single-use final-test protocol.

## Recruiter verification path

A reviewer can verify the project without restricted data:

1. Install Python 3.12 and run `python -m pip install -e ".[dev]"`.
2. Run `python scripts/public_repo_audit.py`.
3. Run `python -m ruff check src tests scripts app.py`.
4. Run `python -m pytest --cov=anaemia_ml --cov-report=term-missing`.
5. Run `anaemia-day2 --smoke --output-directory runs/day2-smoke`.
6. Run `streamlit run app.py`, or build the Docker image with `docker build -t nfhs5-anaemia-severity-ml .`.

The synthetic smoke path verifies engineering behavior only. The default dashboard view shows disclosure-checked aggregate NFHS development/calibration evidence and labels the locked test as untouched.

## Research boundary

The next research-only milestone is a separately reviewed confirmatory release after the protocol-defined state-held-out, final-test, bootstrap uncertainty and subgroup/geographic robustness gates are actually executed on authorized data. The public engineering release should not be delayed or misrepresented while those scientific gates remain intentionally closed.
