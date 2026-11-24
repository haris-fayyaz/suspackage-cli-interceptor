# Experiments

All numbers come from `reports/metrics.json`. Seed 42. Plots are in `reports/figures/`.

## Setup
- Data: 26,959 usable package versions (12,291 malicious, 14,668 benign). 906 rows failed to read and are excluded.
- Inputs: 32 static features plus 11 per-KB density features. Nothing from a package is executed.
- Split groups: rows are linked when they share a family name or an identical feature vector. 8,768 rows have a feature vector identical to another row. This gives 17,258 groups, and the largest holds 210 rows (0.8%).
- Training rows are capped at 20 per group. Test rows are never capped.
- Tuning: 20 random draws each for Random Forest and XGBoost, 5 grouped folds, scored by average precision.
- Tracking: MLflow experiment `suspackage-training`, 13 runs, parent run `training`.

## Three views
- Holdout: 20% of groups (5,394 rows, 2,459 malicious) never touched during tuning. The other 80% is dev, used for 5-fold grouped CV.
- Cross-dataset: train on pypi_malregistry malware plus benign groups, test on DataDog malware plus held-out benign groups. 144 groups shared by both malware sets are removed from training.
- Ablations: the best model refit with feature groups removed.

## Holdout results
Raw scores at threshold 0.5.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Brier |
|---|---|---|---|---|---|---|---|
| Dummy baseline | 0.000 | 0.000 | 0.000 | 0.500 | 0.456 | 0.000 | 0.253 |
| Logistic regression | 0.949 | 0.943 | 0.946 | 0.985 | 0.985 | 0.043 | 0.039 |
| Random forest | 0.991 | 0.944 | 0.967 | 0.994 | 0.994 | 0.007 | 0.024 |
| XGBoost | 0.987 | 0.948 | 0.967 | 0.994 | 0.994 | 0.011 | 0.024 |

- Cross-validated PR-AUC on dev: random forest 0.9926, XGBoost 0.9921, logistic regression 0.9827.
- Random forest and XGBoost are almost tied. Random forest won by 0.0005 on CV, so it is the shipped model.
- Recall at 1% FPR: random forest 0.948, XGBoost 0.946, logistic regression 0.899.

### Per source (holdout, threshold 0.5)

| Model | DataDog recall | malregistry recall | top_pypi FPR | pypi_random FPR |
|---|---|---|---|---|
| Logistic regression | 0.958 | 0.941 | 0.030 | 0.082 |
| Random forest | 0.900 | 0.950 | 0.003 | 0.021 |
| XGBoost | 0.912 | 0.954 | 0.005 | 0.027 |

- Sample sizes: DataDog 331, malregistry 2,128, top_pypi 2,202, pypi_random 733.
- Random benign packages are flagged more often than popular ones. For random forest it is about 7 times more often (2.1% against 0.3%).

### Duplicates matter
- Many holdout rows repeat the same feature vector. Keeping one row per vector leaves 3,663 rows, and the malicious share falls to about 27% (the dummy PR-AUC of 0.272).
- Random forest: recall 0.944 on all rows, 0.909 on unique vectors. PR-AUC 0.994 and 0.982.
- Quote the unique-vector numbers when you want the cautious figure. Both are in `metrics.json` as `all` and `dedup`.

## Cross-dataset results
Train 18,218 rows after the cap. Test 4,864 rows, 1,866 of them DataDog malware. 2,155 pypi_malregistry rows were dropped for sharing groups with DataDog.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Recall at 1% FPR |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.938 | 0.931 | 0.935 | 0.971 | 0.974 | 0.038 | 0.848 |
| Random forest | 0.983 | 0.920 | 0.950 | 0.986 | 0.985 | 0.010 | 0.921 |
| XGBoost | 0.974 | 0.913 | 0.942 | 0.989 | 0.986 | 0.015 | 0.893 |

- Random forest on DataDog: recall 0.920. The holdout recall on DataDog was 0.900 (331 rows), so we saw no large drop.
- On unique vectors only (3,692 rows, 1,022 malicious), random forest recall is 0.915 and PR-AUC 0.981.
- The dummy baseline is not run here.

## Ablations (random forest, holdout)

| Variant | Inputs | PR-AUC | Recall | FPR | DataDog recall | pypi_random FPR |
|---|---|---|---|---|---|---|
| full | 43 | 0.9942 | 0.944 | 0.007 | 0.900 | 0.021 |
| no name features | 37 | 0.9950 | 0.949 | 0.009 | 0.903 | 0.023 |
| no metadata features | 38 | 0.9908 | 0.939 | 0.011 | 0.879 | 0.025 |
| no sp_present and no metadata | 37 | 0.9909 | 0.938 | 0.009 | 0.894 | 0.019 |
| raw counts, no densities | 32 | 0.9932 | 0.931 | 0.009 | 0.897 | 0.019 |

- Name features add nothing. PR-AUC is slightly higher without them.
- Metadata is worth about 0.003 PR-AUC. Removing `sp_present` as well changes nothing more.
- Densities add 0.001 PR-AUC and 0.013 recall.
- The model does not collapse without packaging-style features, so we kept all features.

## Permutation importance (holdout, drop in average precision)
- `x_py_kb` 0.023 (shuffling it changes every density feature at once)
- `meta_has_homepage` 0.011
- `c_subprocess` 0.010
- `meta_n_deps` 0.005
- `obfuscation_score` 0.004
- `c_http` 0.003
- Package size (through densities) and two metadata features rank in the top four. The ablations show they are not the whole story, but they do carry signal.

## Calibration and thresholds
- Scores are calibrated with isotonic regression fitted on out-of-fold predictions.
- Holdout Brier score: 0.0240 raw, 0.0232 calibrated.
- Thresholds are chosen from calibrated out-of-fold scores on all data, never from a test set.

| Policy | Warn at | Block at | Block FPR target | OOF block FPR | OOF block recall |
|---|---|---|---|---|---|
| lenient | 0.298 | 0.829 | 0.5% | 0.37% | 0.935 |
| balanced | 0.174 | 0.667 | 1% | 0.57% | 0.942 |
| strict | 0.133 | 0.298 | 3% | 2.80% | 0.963 |

- The warn FPR targets (3%, 5%, 10%) are our choice. The block targets come from the project brief.
- Check on the holdout, using thresholds chosen on dev only: balanced blocks 94.4% of malware at 0.72% FPR. Lenient 93.4% at 0.34%. Strict 95.9% at 1.91%.

## Final model
- Random forest, 500 trees, max depth 30, `max_features` sqrt, min samples per leaf 1.
- Fit on 23,750 rows (all usable data after the 20-per-group cap), all features and densities, calibrated.
- Calibrated out-of-fold results: precision 0.983, recall 0.955, F1 0.969, ROC-AUC 0.992, PR-AUC 0.9925, FPR 1.4%, Brier 0.0228.
- By source: DataDog recall 0.945, malregistry 0.956, top_pypi FPR 0.7%, pypi_random FPR 3.4%.
- On unique vectors: recall 0.922, PR-AUC 0.977.

## Decisions
- Split by merged groups, not family names alone, because 32% of rows have a twin.
- Pick the model by cross-validation, never by the holdout.
- No SMOTE. Classes are close to balanced, so class weights are enough.
- CodeBERT is future work. The raw archives it needs are gone.
- Keep all features. The ablations did not justify dropping any.
