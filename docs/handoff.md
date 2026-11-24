# Handoff for the CLI team

## Get the model
```powershell
git clone https://dagshub.com/haris-fayyaz/suspackage-cli-interceptor.git
cd suspackage-cli-interceptor
git checkout v0.1-model
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
dvc pull
```
- `git rev-parse v0.1-model` prints the commit hash.
- `dvc pull` downloads `models/suspackage_model.joblib` and `data/raw/top-pypi-packages.csv`.
- Exact package versions are in `requirements.lock.txt`. Training used Python 3.14.7.

## Use it
Flow: archive, `featurize()`, `ModelBundle.predict_proba()`, `decide()`.
```python
import pandas as pd
from suspackage.config import load_params
from suspackage.extract import featurize
from suspackage.features.name import NameFeatures, norm
from suspackage.model_bundle import ModelBundle

params = load_params("params.yaml")
popular = (pd.read_csv(params["paths"]["popular"]).sort_values("download_count", ascending=False)["project"]
           .map(norm).head(params["extract"]["popular_top_n"]).tolist())
namer = NameFeatures(popular)                      # build once, reuse
bundle = ModelBundle.load("models/suspackage_model.joblib")  # load once, reuse

features = featurize(name, archive_path, params["extract"], namer)
score = bundle.predict_proba(features)             # 0 to 1 risk
action = bundle.decide(score, "balanced")          # "allow", "warn" or "block"
```
- This snippet was tested on a small test archive and produced no missing-feature warnings.
- Pass the whole dict from `featurize()`. Extra keys are ignored. Missing keys default to 0 and log a warning.
- `x_py_kb` must be in the dict. The model uses it to compute per-KB densities.
- Policies: `lenient`, `balanced`, `strict`. Thresholds are in `models/thresholds.json`.

| Policy | Warn at | Block at |
|---|---|---|
| lenient | 0.298 | 0.829 |
| balanced | 0.174 | 0.667 |
| strict | 0.133 | 0.298 |

- A score at or above Block is blocked. At or above Warn is a warning. Below Warn is allowed.
- Start with `balanced`.

## Integration rules
- Name not on PyPI (JSON API returns 404): skip the model and flag it as a slopsquatting candidate.
- Typosquat rule (edit distance to the top-1,000 popular names) runs separately from the model.
- Wheel-only package: `featurize()` raises `ValueError`. Treat it as `warn`.
- The model scores only packages that exist and have a source archive.
- Never run or import anything from the package. All features are read statically.

## What the model scored
- Holdout (random forest, raw score at 0.5): precision 0.991, recall 0.944, F1 0.967, ROC-AUC 0.994, FPR 0.7%.
- Cross-dataset (train malregistry, test DataDog): recall 0.920, FPR 1.0%, PR-AUC 0.985.
- Balanced policy on the holdout: blocks 94.4% of malware at 0.72% FPR.
- Full tables are in `docs/experiments.md`. Read `docs/limitations.md` before you quote these in the report.

## Where things are
- MLflow experiment: https://dagshub.com/haris-fayyaz/suspackage-cli-interceptor.mlflow/#/experiments/0
- Parent run `training`: https://dagshub.com/haris-fayyaz/suspackage-cli-interceptor.mlflow/#/experiments/0/runs/c4bba2409eaa41f0ae84a5393efbe618
- Registered model: `suspackage-risk-model`, version 1.
- Metrics: `reports/metrics.json`. Plots: `reports/figures/`. Model card: `models/model_card.md`.
- Settings and seeds: the `train:` section of `params.yaml`. Seed 42.

## Retrain
```powershell
$env:MLFLOW_TRACKING_USERNAME="<dagshub username>"
$env:MLFLOW_TRACKING_PASSWORD="<dagshub token>"
dvc repro
dvc push
```
- It takes about 8 minutes on a Ryzen 7 5800H.
- Each run adds 13 MLflow runs and a new model version.
- Never put the token in Git or in code.
