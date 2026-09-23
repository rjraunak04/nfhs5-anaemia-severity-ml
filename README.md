# NFHS-5 Anaemia Severity ML

![CI](https://github.com/rjraunak04/nfhs5-anaemia-severity-ml/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/status-portfolio%20ready-16a34a)

**Survey-aware multiclass ML with leakage-safe validation, calibrated probabilities, explainability, and a policy-gated research copilot.**

**[Open the live dashboard](https://nfhs5-anaemia-severity-ml-production.up.railway.app)** · [Results](docs/development_results.md) · [Model card](docs/model_card.md) · [Research protocol](docs/research_protocol.md)

## At a glance

| | |
|---|---|
| **Problem** | 4-class anaemia severity prediction: None / Mild / Moderate / Severe |
| **Scale** | 724,115 NFHS-5 records across 36 States/UTs and 707 districts |
| **Validation** | PSU-disjoint 70/10/20 split + grouped nested cross-validation |
| **Selected development model** | Random Forest · macro-F1 0.325 · severe recall 0.125 |
| **Engineering** | 227 tests · CI quality gates · Docker · Railway |
| **Agent layer** | Policy-gated copilot · 20/20 golden eval cases · external LLM fallback disabled by default |

## Why this project

NFHS-5 is not an ordinary tabular dataset: observations are clustered by survey design, the target is imbalanced, and some variables can leak outcome information.

I built the project around those constraints instead of treating them as cleanup steps. The pipeline keeps PSUs disjoint across partitions, fits preprocessing inside training folds, calibrates probabilities on a separate split, reports aggregate SHAP only, and keeps the final test locked behind an explicit research protocol.

## Dataset and target

| Item | Value |
|---|---|
| Survey | NFHS-5 India, 2019–2021 |
| Analytic records | 724,115 |
| Coverage | 36 States/UTs, 707 districts |
| Source variables | 40 |
| Target | None / Mild / Moderate / Severe anaemia |
| Development / calibration / locked test | 70% / 10% / 20% by PSU |

The primary label comes from NFHS `v457`. Adjusted haemoglobin (`v456`) is used only for outcome audit/sensitivity work and is never a predictor. Raw NFHS/DHS microdata are not distributed in this repository.

## Pipeline

```mermaid
flowchart LR
    A[NFHS-5 restricted extract] --> B[Schema + quality checks]
    B --> C[PSU-disjoint 70/10/20 split]
    C --> D[Fold-local preprocessing]
    D --> E[Grouped nested CV]
    E --> F[Model selection]
    F --> G[Calibration split]
    G --> H[Aggregate SHAP]
    H --> I[Streamlit dashboard]
    G -. frozen protocol .-> J[Locked final test]
```

Key design choices:

- composite PSU grouping prevents cluster overlap between partitions;
- preprocessing is fitted inside training folds only;
- numeric features use median imputation plus missingness indicators;
- categorical missingness is represented explicitly;
- model selection uses grouped nested cross-validation;
- probability calibration is performed on a separate calibration partition;
- the final test remains locked until the confirmatory research protocol is executed.

## Results — development only

These are **development/calibration estimates, not final-test results**.

| Model | Macro-F1 | Balanced accuracy | Severe recall |
|---|---:|---:|---:|
| Multinomial logistic regression | 0.270 ± 0.002 | 0.332 | 0.452 |
| Random forest | **0.325 ± 0.002** | 0.332 | 0.125 |
| LightGBM | 0.312 ± 0.001 | 0.360 | 0.368 |

Random forest was selected by the prespecified primary metric (macro-F1). Its low severe-class recall is an important limitation and is one reason this project is presented as a research/engineering system rather than a clinical screening tool.

Calibration on the held-out calibration partition improved:

- weighted log loss: **1.252 → 1.211**
- multiclass Brier score: **0.690 → 0.678**
- expected calibration error: **0.066 → 0.013**

Aggregate SHAP analysis highlighted BMI, age, and education among the leading model features. These are model associations, not causal effects.

## Engineering highlights

- **Leakage-safe ML:** strict predictor exclusions and fold-local transformations
- **Survey-aware validation:** PSU-disjoint splits and grouped nested CV
- **Reproducibility:** configuration/data fingerprints and resumable checkpoints
- **Model governance:** locked-test guardrails and explicit release boundaries
- **Public-data safety:** CI audit blocks restricted data, serialized models, secrets, and row-level outputs
- **Deployment:** Dockerized Streamlit app running on Railway
- **Quality gates:** Ruff, pytest/coverage, notebook cleanliness checks, Docker build

## Research Copilot

The live dashboard also includes a small, read-only agent layer for model comparison, calibration, SHAP, release readiness, and next-experiment planning.

`request → deterministic router → optional LLM fallback → approved tool → policy gate → traced response`

- known requests stay local and deterministic;
- every action is restricted to a closed intent set;
- external fallback is optional and disabled in production by default;
- the external planner never receives NFHS rows, model evidence, or validation artifacts;
- 20 version-controlled golden cases are enforced in CI.

The copilot is deliberately separated from the ML pipeline: it can explain and orchestrate approved read-only actions, but it cannot unlock the final test or make clinical decisions.

## Tech stack

**Python · pandas · scikit-learn · LightGBM · Optuna · SHAP · Pydantic · Streamlit · Docker · GitHub Actions · Railway**

## Repository structure

```text
.
├── app.py                 # public Streamlit dashboard
├── configs/               # data + validation contracts
├── data/README.md         # restricted-data handling
├── demo/                  # disclosure-checked aggregate demo assets
├── docs/                  # protocol, results, model card, engineering notes
├── notebooks/             # clean research notebook
├── scripts/               # public-repository safety audit
├── src/anaemia_ml/
│   ├── agents/            # typed agent planner, tools and governance
│   └── ...                # reusable ML pipeline
├── tests/                 # unit/integration/leakage tests
├── Dockerfile
└── pyproject.toml
```

## Run locally

Python 3.12:

```bash
git clone https://github.com/rjraunak04/nfhs5-anaemia-severity-ml.git
cd nfhs5-anaemia-severity-ml

python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/public_repo_audit.py
python -m pytest -q
streamlit run app.py
```

Docker:

```bash
docker build -t nfhs5-anaemia-severity-ml .
docker run --rm -p 8501:8501 nfhs5-anaemia-severity-ml
```

## Documentation

- [Development results](docs/development_results.md) — model comparison, calibration and aggregate SHAP evidence
- [Model card](docs/model_card.md) — intended use, limitations and responsible-use boundary
- [Research protocol](docs/research_protocol.md) — prespecified statistical and validation design
- [Engineering notes](docs/engineering.md) — CI, reproducibility, deployment and public-data safeguards

## Scope and responsible use

This is a research and ML-engineering project, not a diagnostic service. It must not replace haemoglobin testing or guide treatment. The locked final test, cluster-bootstrap confidence intervals, subgroup/geographic robustness and external validation remain separate confirmatory research steps.

## Author

**Ankur Kumar Jaiswal** · [GitHub](https://github.com/rjraunak04)

NFHS/DHS participant-level data remain governed by the data provider's access agreement. Public visibility of this repository does not grant rights to redistribute the underlying survey data.
