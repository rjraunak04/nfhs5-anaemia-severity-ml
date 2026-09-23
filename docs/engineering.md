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


## Agentic research copilot

The copilot is intentionally built on top of the existing ML package instead of duplicating statistical logic inside prompts.

Architecture:

```text
User question
    ↓
HybridPlanner
    ├── high-confidence RuleBasedPlanner
    └── optional external LLM fallback
    ↓
Closed AgentIntent enum
    ↓
Approved deterministic tool
    ↓
Disclosure / governance policy
    ↓
Execution trace
    ↓
Structured response + evidence source + warnings
```

The live deployment defaults to `HybridPlanner` without an external fallback. Known requests therefore stay local and deterministic. `CallablePlanner` is the model-agnostic adapter for a future LLM provider; when configured, it is consulted only after deterministic routing cannot resolve the request. Its output must still validate against the closed intent enum.

V2 adds two automation actions: `release_readiness` audits public engineering gates from aggregate evidence, while `next_experiment` produces an ordered protocol-safe research plan. Both are read-only. They cannot retrain models, mutate research artifacts or open the locked final test.

Every response carries a safe trace with four concepts: planner decision, policy checks, deterministic tool, and grounded response. This gives the agent observable behavior without exposing hidden reasoning or respondent-level data.

This separation is intentional: the planner decides **what approved action to request**, deterministic Python tools decide **how project evidence is read**, and governance policies decide **whether the action is allowed**. The result is easier to test, explain and extend without giving an LLM direct authority over restricted data or scientific release gates.
