"""Write models/model_card.md from real run numbers. Nothing here is hand-typed metrics."""
from __future__ import annotations

from pathlib import Path


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _row(name: str, m: dict) -> str:
    return (f"| {name} | {_f(m.get('precision'))} | {_f(m.get('recall'))} | {_f(m.get('f1'))} | "
            f"{_f(m.get('roc_auc'))} | {_f(m.get('pr_auc'))} | {_f(m.get('false_positive_rate'))} | {_f(m.get('brier_score'))} |")


def write_model_card(path: Path, info: dict, metrics: dict) -> None:
    """info: model, params, n_train, data_hash, thresholds, columns. metrics: contents of metrics.json."""
    a = metrics.get("holdout", {})
    b = metrics.get("cross_dataset", {})
    head = "| Setting | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Brier |\n|---|---|---|---|---|---|---|---|"
    rows = []
    best = a.get("best_model")
    if best and best in a.get("models", {}):
        rows.append(_row(f"Group holdout ({best}, raw score at 0.5)", a["models"][best]["holdout"]["all"]))
    for k, v in b.get("models", {}).items():
        if k == best:
            rows.append(_row(f"Cross-dataset: train malregistry, test DataDog ({k})", v["all"]))
    if info.get("oof"):
        rows.append(_row("Final model, calibrated out-of-fold (5 folds)", info["oof"]))
    thr = info["thresholds"]
    thr_rows = "\n".join(
        f"| {n} | {t['warn']:.3f} | {t['block']:.3f} | {t['block_fpr_target']:.3f} | "
        f"{t['oof_block_fpr']:.4f} | {t['oof_block_recall']:.3f} |" for n, t in thr.items())
    text = f"""# SusPackage model card

## What it is
- A risk model for Python packages that exist on PyPI. It scores one package from 0 to 1.
- Model: {info['model']} with isotonic calibration. Trained on {info['n_train']} rows.
- Training data version: `{info['data_hash']}`
- Input: {len(info['columns'])} columns from `featurize()`, in the order of `feature_columns.json`.
- Nothing from a package is executed. Features are read from the archive as text.

## What it is not
- It does not handle names that are missing from PyPI. Those are flagged as slopsquatting by a rule, with no model.
- It cannot score wheel-only packages. The CLI should warn on those.
- The typosquat name rule runs separately. The name features here are weak.

## Results
{head}
{chr(10).join(rows) if rows else '| no results found | | | | | | | |'}

- Details, per-source numbers and ablations are in `reports/metrics.json` and `docs/experiments.md`.

## Policies
- A score at or above `block` is blocked. At or above `warn` is a warning. Below that is allowed.
- Thresholds come from calibrated out-of-fold scores, never from the test sets.

| Policy | Warn at | Block at | Block FPR target | OOF block FPR | OOF block recall |
|---|---|---|---|---|---|
{thr_rows}

## Known limits
- Malware labels come mostly from one detector (GuardDog) and some samples are proof-of-concept packages.
- Benign random packages were not checked by hand.
- Packaging style differs between sources, so the model can learn "plain package means malicious". See the ablations.
- DataDog and pypi_malregistry look different, so expect a drop on unseen malware styles.
- Obfuscated or novel malware can score low. Treat the score as a screen, not a verdict.

## Loading it
- `ModelBundle.load("models/suspackage_model.joblib")`, then `predict_proba(features)` and `decide(score, policy)`.
- The file is a joblib pickle. Load only files built by this project.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
