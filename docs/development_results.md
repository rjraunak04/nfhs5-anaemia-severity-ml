# NFHS-5 development and calibration results

**Status:** Development evidence only. The protocol's locked-test partition has not been evaluated. No final performance or clinical-use claim is supported.

## Provenance

The restricted 40-column NFHS-5 extract passed the data contract: 724,115 rows, 36 states/UTs, 707 districts, zero duplicate respondent rows and zero validation errors. One nonblocking profile warning remains: 2,770 observed composite strata versus the 2,710 reference value. The local source dataset SHA-256 is `fde108e65c691cb270e68fe6fd2e1fa72755e671dc419dfdb823c1568be75fd0`; experiment fingerprint is `fc9961f8549ec891eed23ffd3b5608b990a7a1fed944825a9b23d4ac41b99faa`. Run code version: `c5ca043`.

PSU-disjoint partition sizes: 506,883 development, 72,412 calibration and 144,820 locked test. The 144,820 test rows remain unevaluated. Each candidate completed five outer folds of grouped nested cross-validation.

## Development model comparison

| Model | Outer-fold macro-F1 mean ± SD | Balanced accuracy | Severe recall |
|---|---:|---:|---:|
| Multinomial logistic regression | 0.270 ± 0.002 | 0.332 | 0.452 |
| Random forest | **0.325 ± 0.002** | 0.332 | **0.125** |
| LightGBM | 0.312 ± 0.001 | 0.360 | 0.368 |

The prespecified primary metric, macro-F1, selected random forest. Severe recall is substantially lower than the other candidates, so the chosen model should not be described as a suitable severe-anaemia screening tool. These are grouped development estimates; they are not an external or final test score. No uncertainty interval is available yet.

## Probability calibration

The calibration-only, five-fold PSU-disjoint temperature search accepted temperature **0.5**. On the cross-fitted calibration selection estimate, survey-weighted log loss fell from **1.252** to **1.211**, multiclass Brier score from **0.690** to **0.678**, and expected calibration error from **0.066** to **0.013**. These values describe calibration selection on its own partition, not unseen-test generalization.

SHAP was aggregated over 200 calibration observations with a 50-row background. The leading aggregate predictor was BMI (`v445`); age (`v012`) and education (`v133`) also ranked highly. The reported SHAP values describe model associations; they do not establish causes, clinical significance or subgroup performance. No row-level attributions are published.

## Reproduce and inspect

Authorized researchers with the matching extract can run `anaemia-day2 --data <local-path> --output-directory runs/day2-nfhs --code-version c5ca043` after installing the development stack. Training is expensive and writes resumable checkpoints. The public app consumes only [the aggregate summary](../demo/nfhs_development_summary.json); the private fitted `.joblib` files, participant rows and checkpoint files are excluded from the repository. The calibrated artifact manifest recorded SHA-256 `bfc4ceaa296935436aab051f719bcf927309e50706c42e8a6c42e1ea4c283210`; the artifact itself is not published.

Next research gates: prespecified locked-test evaluation, cluster-bootstrap intervals, subgroup and geographic robustness, and independent clinical review.
