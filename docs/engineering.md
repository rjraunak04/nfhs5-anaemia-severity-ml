# Engineering notes

This document collects the operational details that support the public release without cluttering the project README.

## Release status

The software and portfolio release is complete: the Python package, validation contracts, grouped model-selection workflow, calibration stage, aggregate SHAP reporting, tests, CI, Docker image and public Streamlit dashboard are implemented.

The scientific confirmatory release is intentionally separate. The locked final-test partition has not been evaluated, so this repository does not claim final clinical performance, subgroup fairness, geographic robustness or external validation.

## Quality checks

Run the same checks used in CI:

```bash
python scripts/public_repo_audit.py
python -m ruff check src tests scripts app.py
python -m pytest --cov=anaemia_ml --cov-report=term-missing
docker build -t nfhs5-anaemia-severity-ml .
```

GitHub Actions additionally verifies that committed notebooks contain no outputs or execution counts and that their code cells parse successfully.

## Public repository safeguards

The public audit fails CI when tracked files contain common secret formats, local Windows user paths, restricted data/model file types, generated artifact directories, or row-level dashboard keys.

The public repository must not contain:

- NFHS/DHS respondent-level data;
- respondent or PSU identifiers;
- row-level predictions/probabilities;
- fitted model binaries or checkpoints;
- environment files, API keys or tokens.

The dashboard reads only disclosure-checked aggregate JSON and a clearly separated synthetic engineering demo.

## Deployment

Production dashboard:

https://nfhs5-anaemia-severity-ml-production.up.railway.app

Railway deploys the root `Dockerfile` from `main`. Streamlit binds to Railway's injected `PORT`, with `/_stcore/health` configured as the service healthcheck.

For a local container:

```bash
docker build -t nfhs5-anaemia-severity-ml .
docker run --rm -p 8501:8501 nfhs5-anaemia-severity-ml
```

## Development workflow

Changes should be made on focused branches and merged only after CI passes. Squash merging keeps `main` readable. Recommended commit prefixes are `feat:`, `fix:`, `test:`, `docs:`, `ci:` and `chore:`.

Any change to eligibility, outcome definitions, predictors, splitting, calibration or evaluation should be documented because those choices affect the scientific protocol.

## Reproducibility boundary

Long-running research runs record dataset/configuration fingerprints and model/output checksums. Incompatible checkpoints must not be reused.

A future confirmatory research release requires the frozen protocol-defined state-held-out analysis, single-use locked-test evaluation, PSU-within-strata bootstrap uncertainty, subgroup/geographic robustness checks and final manuscript reporting.
