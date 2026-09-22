# Model card: NFHS-5 anaemia severity development candidate

## Intended purpose

Research prototype for comparing four-class anaemia severity associations in women aged 15–49 in the validated NFHS-5 India analytic extract. It is a portfolio demonstration of survey-aware ML engineering. **It is not a diagnostic tool**, individual risk service, replacement for haemoglobin testing, or basis for treatment and resource allocation.

## Data and labels

The restricted extract has 724,115 rows and 40 selected source fields. NFHS `v457` codes map to None (0), Mild (1), Moderate (2), Severe (3). Adjusted haemoglobin (`v456`), outcome (`v457`), participant identifiers, weights and sampling identifiers are excluded from predictors. The India-policy variant uses 32 predictors and can include state. Survey weights enter evaluation; PSU groups separate partitions and validation folds. The dataset, fitted models and respondent-level outputs are not in the public repository.

## Training and selection

The pipeline compares multinomial logistic regression, random forest and LightGBM using nested PSU-grouped development cross-validation, training-fold preprocessing and class weighting. Macro-F1 selected random forest (`n_estimators=400`, `min_samples_leaf=8`, `max_depth=None`). Its probabilities were then temperature-scaled (`T=0.5`) using the separate calibration partition and group-disjoint cross-fitting. See [development results](development_results.md) and the [protocol](research_protocol.md).

## Performance status and limitations

Development outer-fold macro-F1 is 0.325 (SD 0.002); severe recall is 0.125. Calibration selection estimated log loss of 1.211 after scaling, versus 1.252 before; calibration ECE changed from 0.066 to 0.013. These are development/calibration estimates only. The locked-test set (144,820 rows) is unevaluated. Severe recall is poor for a potential screening application. No confidence intervals, threshold optimization, geographic robustness, prospective validation, or subgroup fairness assessment has been reported. Cross-sectional survey predictors cannot establish causes.

## Explainability and safeguards

Aggregate SHAP describes associations for 200 calibration rows, without exposing individual attributions. Predictor ranking must not be read as intervention benefit. The public dashboard reads aggregate JSON only and must keep the synthetic engineering demo visibly separate from the real development summary. A model artifact checksum is recorded in local manifests, but the fitted model is not public. The reported validation warning (observed 2,770 strata against a 2,710 reference) should be investigated before paper-grade conclusions.

## Release policy

No restricted data, identifiable rows, model weights, local paths, secrets, or respondent-level predictions belong in Git or a hosted app. Before any clinical or final performance claim, complete the preregistered locked-test, uncertainty, subgroup and governance gates. The deployment is an **aggregate results dashboard**, not a patient prediction service.
