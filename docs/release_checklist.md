# Day 3 release checklist

This checklist separates completed engineering/recruiter-release work from research gates that must remain closed until the prespecified protocol is executed.

## Completed for the development release

- [x] Restricted-data governance and leakage-safe data contract
- [x] PSU-disjoint development, calibration and locked-test partitions
- [x] Fold-local preprocessing and imbalance controls
- [x] Grouped nested cross-validation for candidate selection
- [x] Resumable training workflow and integrity-checked artifacts
- [x] Calibration-only temperature selection with PSU-disjoint cross-fitting
- [x] Aggregate-only SHAP reporting
- [x] Disclosure-checked NFHS development summary for the dashboard
- [x] Clearly separated synthetic engineering demo
- [x] Streamlit recruiter/research dashboard
- [x] Model card and development-results report
- [x] Deployment configuration and root requirements file
- [x] Unit/integration tests and Ruff quality gate, including app.py
- [x] Citation metadata
- [x] Raw NFHS/DHS data, respondent predictions, checkpoints and model binaries excluded from Git

## Intentionally not claimed in this release

- [ ] Locked-test performance
- [ ] PSU-within-strata bootstrap confidence intervals
- [ ] Subgroup fairness/robustness conclusions
- [ ] State-held-out or external validation conclusions
- [ ] Clinical screening or diagnostic utility
- [ ] Causal interpretation of SHAP or predictor associations
- [ ] Final manuscript result tables, DOI or publication claim

## Before making the repository or dashboard public

1. Confirm the repository contains no restricted NFHS/DHS records, secrets, local paths, model binaries or row-level outputs.
2. Confirm the dashboard headline says the NFHS evidence is development/calibration only and the locked test is untouched.
3. Run `python -m ruff check src tests app.py`.
4. Run `python -m pytest --cov=anaemia_ml --cov-report=term-missing`.
5. Start `streamlit run app.py` and verify all four tabs plus the synthetic-data switch.
6. Review repository visibility and licensing separately; do not assume a software license until one is intentionally selected.
7. Deploy only the aggregate dashboard described in `docs/deployment.md`.
8. Add a verified live URL to the README/repository homepage only after the deployed app passes the same disclosure checks.

## Research-release gate

A future final research release may be tagged only after the locked-test, uncertainty, subgroup/geographic robustness and reproducibility checks are complete. Until then, this repository should be described as a development research/engineering release.
