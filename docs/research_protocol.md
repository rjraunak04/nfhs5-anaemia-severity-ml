# Research Protocol

## Project title

Survey-Aware Machine Learning for Multiclass Anaemia Severity
Classification Among Women of Reproductive Age in India Using NFHS-5

## Protocol status

- Study type: Secondary cross-sectional analysis
- Data source: India NFHS-5, 2019–2021
- Population: Women aged 15–49 years
- Protocol version: 0.1.0
- Status: Prespecified before final model training
- Primary analysis dataset: 40-column restricted NFHS-5 extract

## 1. Background

Anaemia remains an important public-health concern among women of
reproductive age in India. Nationally representative household surveys
provide an opportunity to assess whether demographic, socioeconomic,
reproductive, nutritional, environmental and healthcare-access factors
can support population-level anaemia severity classification.

This study develops and evaluates survey-aware machine-learning models
for four-class anaemia severity classification using NFHS-5 data.

The study is cross-sectional. Results will not be interpreted as causal,
prospective or as a replacement for haemoglobin testing or clinical
diagnosis.

## 2. Primary research question

How accurately can leakage-safe demographic, socioeconomic,
reproductive, nutritional, anthropometric, environmental and
healthcare-access variables classify anaemia severity among women aged
15–49 years in India while accounting for the NFHS-5 complex survey
design?

## 3. Study objectives

### Primary objective

Develop and internally validate a four-class anaemia severity
classification model using PSU-grouped nested cross-validation.

### Secondary objectives

1. Evaluate severe-anaemia discrimination and recall.
2. Evaluate probability calibration.
3. Quantify uncertainty using survey-cluster bootstrap confidence intervals.
4. Assess performance across demographic and socioeconomic subgroups.
5. Assess geographic robustness using state-held-out validation.
6. Compare survey-weighted and unweighted performance.
7. Compare class-weighting and fold-local SMOTENC strategies.
8. Explain held-out model predictions using class-specific SHAP values.
9. Assess transportability using models that exclude state and district
   from the predictor set.

## 4. Study population

### Inclusion criteria

- Women aged 15–49 years included in the NFHS-5 Individual Recode data.
- Valid anaemia severity outcome in `v457`.
- Valid survey weight in `v005`.
- Valid PSU or cluster identifier in `v021`.
- Valid state identifier in `v024`.
- Availability of the required survey-design variables.

### Exclusion criteria

- Missing or invalid primary outcome.
- Invalid survey weight.
- Missing PSU required for grouped validation.
- Records failing prespecified integrity checks.

Participants will not be excluded solely because predictor values are
missing. Predictor missingness will be handled inside model-training
folds.

A participant-flow table will report the number of records at every
inclusion and exclusion stage.

## 5. Outcome definition

The primary outcome is the NFHS-5 anaemia-level variable `v457`.

Raw outcome labels will be mapped to:

| Model code | Outcome class |
|---:|---|
| 0 | No anaemia |
| 1 | Mild anaemia |
| 2 | Moderate anaemia |
| 3 | Severe anaemia |

Mapping will be performed using verified NFHS value labels rather than
unverified numeric assumptions.

The adjusted haemoglobin variable `v456` will be used only for outcome
audit and sensitivity analysis. It will never be used as a predictor.

Pregnancy-trimester-specific haemoglobin sensitivity analysis will not
be claimed because pregnancy-duration variable `v214` is unavailable
in the selected dataset.

## 6. Predictor domains

The candidate predictor set is defined before model development and
includes:

- Age and residence
- Marital status
- Religion and caste
- Wealth and completed education
- Fertility and recent-birth history
- Current pregnancy
- Pregnancy-termination history
- Contraceptive use
- Breastfeeding and amenorrhea
- Body mass index
- Healthcare-access barriers
- Drinking water and sanitation
- Cooking fuel
- Health insurance
- Diabetes indicator
- Food-frequency variables

The following variables are not predictors:

- `v002`, `v003`: respondent/household identification
- `v005`: survey weight
- `v021`: PSU/grouping
- `v022`: survey strata
- `sdist`: district validation group
- `v456`: adjusted haemoglobin
- `v457`: target

The primary India policy model may include state. The transportability
model will exclude state and district information.

## 7. Survey design

Survey weights will be calculated as:

`survey_weight = v005 / 1,000,000`

Composite identifiers will be created when necessary:

- PSU: state combined with `v021`
- Stratum: state combined with `v022`

Survey weights, PSUs and strata will be used for descriptive analysis,
performance evaluation and uncertainty estimation.

Weighted and unweighted model-training analyses will be compared as a
prespecified sensitivity analysis.

## 8. Data splitting

All observations from the same composite PSU will remain in the same
partition.

Planned partitions:

- Development set: 70% of PSUs
- Calibration set: 10% of PSUs
- Locked final test set: 20% of PSUs

The split will seek reasonable outcome and geographic balance without
placing the same PSU in multiple partitions.

The final test set will not be used for:

- Feature selection
- Hyperparameter tuning
- Model selection
- Calibration-method selection
- Ensemble construction
- Decision-threshold selection

## 9. Model development

### Baselines

- Majority-class dummy classifier
- Multinomial logistic regression
- Ordinal regression

### Machine-learning candidates

- Random forest
- LightGBM
- XGBoost
- CatBoost

An ensemble will be retained only when its construction is based
entirely on development/calibration data and it provides a prespecified
improvement.

## 10. Preprocessing and leakage prevention

For every cross-validation fold, the following operations will be
fitted using fold-training data only:

1. Missing-value imputation
2. Categorical encoding
3. Numeric transformation or scaling
4. Feature selection, if used
5. Class-imbalance treatment
6. Model fitting
7. Probability calibration where applicable

No imputation, scaling, encoding or synthetic oversampling will be
performed on the complete dataset before cross-validation.

Class weighting will be the primary imbalance strategy. SMOTENC will
be evaluated as a separate fold-local sensitivity analysis. SMOTENC and
class weighting will not be combined without an explicitly reported
ablation analysis.

## 11. Validation and tuning

Model development will use nested PSU-grouped cross-validation:

- Outer folds: 5
- Inner folds: 3
- Hyperparameter optimization: Optuna
- Optimization data: development set only
- Primary selection metric: macro-F1

Optimization trials and completed folds will be checkpointed.
Hyperparameter search space and computational budget will be frozen
before the final paper run.

## 12. Evaluation metrics

### Primary metric

- Macro-F1

### Key severe-class metrics

- Severe-class recall
- Severe-class precision
- Severe-class F1
- Severe-class area under the precision–recall curve

### Secondary metrics

- Balanced accuracy
- Weighted-F1
- Macro area under the receiver operating characteristic curve
- Multiclass Brier score
- Log loss
- Expected calibration error
- Quadratic-weighted kappa
- Ordinal mean absolute error

Metrics will be reported with survey-weighted and unweighted estimates
where appropriate.

## 13. Probability calibration

Candidate calibration approaches may include:

- Temperature scaling
- Sigmoid calibration
- Isotonic calibration where statistically appropriate

Calibration-method selection will use calibration data or valid
out-of-fold predictions only.

Pre-calibration and post-calibration performance will be reported.

## 14. Uncertainty analysis

Final-test uncertainty will be estimated using 500 cluster-bootstrap
replicates.

PSUs will be resampled within survey strata. Individual synthetic rows
will not be bootstrapped independently.

Two-sided 95% percentile confidence intervals will be reported.

## 15. Explainability

SHAP analysis will be performed on untouched held-out observations.

The analysis will include:

- Global feature importance
- Class-specific absolute SHAP values
- Class-specific signed SHAP values
- Dependence analysis for selected prespecified features

SHAP results will be interpreted as model explanations, not as causal
effects.

## 16. Subgroup and robustness analyses

Prespecified subgroups include:

- Age group
- Pregnancy status
- Residence
- Caste
- Religion
- Wealth
- Education
- State or geographic region
- Breastfeeding status
- Health-insurance status

Overall subgroup metrics will require at least 500 observations.
Severe-class metrics will not be emphasized for subgroups containing
fewer than 50 severe cases.

Subgroup comparisons are exploratory and will not be interpreted as
evidence of discrimination or causal disparity without additional
analysis.

## 17. Sensitivity analyses

Planned sensitivity analyses include:

1. Weighted versus unweighted training.
2. Class weights versus fold-local SMOTENC.
3. Models with and without geographic predictors.
4. Alternative outcome construction using `v456`, where valid.
5. State-held-out geographic validation.
6. Removal of selected engineered interaction features.
7. Performance excluding observations with extensive predictor
   missingness.

Any numerical misclassification-cost matrix will be described as
stakeholder-informed and exploratory, not as prescribed by WHO.

## 18. Reproducibility

Every experiment will record:

- Dataset checksum
- Configuration checksum
- Feature-schema version
- Relevant source-code checksum
- Git commit identifier
- Random seeds
- Package versions
- Completed pipeline stages
- Model and output checksums

Incompatible checkpoints will not be reused.

## 19. Reporting standards

Reporting will follow:

- TRIPOD+AI
- PROBAST+AI
- Relevant DHS survey-analysis guidance
- Applicable journal reporting requirements

## 20. Ethics and responsible use

This study is a secondary analysis of de-identified survey data obtained
under the DHS Program data-use agreement.

The resulting model is intended for research and population-level
analysis. It is not a medical device, diagnostic system or treatment
recommendation tool.

## 21. Protocol deviations

Any change made after model development begins will be documented in
`docs/analysis_deviations.md` with:

- Date
- Git commit
- Original decision
- Revised decision
- Reason for change
- Whether test-set information influenced the change