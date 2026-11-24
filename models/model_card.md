# SusPackage model card

## What it is
- A risk model for Python packages that exist on PyPI. It scores one package from 0 to 1.
- Model: random_forest with isotonic calibration. Trained on 23750 rows.
- Training data version: `cc0edd43212870d6f6d6a360b27da8b5.dir`
- Input: 33 columns from `featurize()`, in the order of `feature_columns.json`.
- Nothing from a package is executed. Features are read from the archive as text.

## What it is not
- It does not handle names that are missing from PyPI. Those are flagged as slopsquatting by a rule, with no model.
- It cannot score wheel-only packages. The CLI should warn on those.
- The typosquat name rule runs separately. The name features here are weak.

## Results
| Setting | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Brier |
|---|---|---|---|---|---|---|---|
| Group holdout (random_forest, raw score at 0.5) | 0.991 | 0.943 | 0.967 | 0.994 | 0.994 | 0.007 | 0.024 |
| Cross-dataset: train malregistry, test DataDog (random_forest) | 0.983 | 0.920 | 0.950 | 0.986 | 0.985 | 0.010 | 0.034 |
| Final model, calibrated out-of-fold (5 folds) | 0.983 | 0.955 | 0.969 | 0.992 | 0.993 | 0.014 | 0.023 |

- Details, per-source numbers and ablations are in `reports/metrics.json` and `docs/experiments.md`.

## Policies
- A score at or above `block` is blocked. At or above `warn` is a warning. Below that is allowed.
- Thresholds come from calibrated out-of-fold scores, never from the test sets.

| Policy | Warn at | Block at | Block FPR target | OOF block FPR | OOF block recall |
|---|---|---|---|---|---|
| lenient | 0.298 | 0.829 | 0.005 | 0.0037 | 0.935 |
| balanced | 0.174 | 0.667 | 0.010 | 0.0057 | 0.942 |
| strict | 0.133 | 0.298 | 0.030 | 0.0280 | 0.963 |

## Known limits
- Malware labels come mostly from one detector (GuardDog) and some samples are proof-of-concept packages.
- Benign random packages were not checked by hand.
- Packaging style differs between sources, so the model can learn "plain package means malicious". See the ablations.
- DataDog and pypi_malregistry look different, so expect a drop on unseen malware styles.
- Obfuscated or novel malware can score low. Treat the score as a screen, not a verdict.

## Loading it
- `ModelBundle.load("models/suspackage_model.joblib")`, then `predict_proba(features)` and `decide(score, policy)`.
- The file is a joblib pickle. Load only files built by this project.
