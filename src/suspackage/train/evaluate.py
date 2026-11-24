"""Metrics, per-source breakdowns and plots."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)


def _nan_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def clean(obj):
    """Make nested metrics JSON safe (NaN to None, numpy to python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _nan_none(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def recall_at_fpr(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Best recall with FPR <= target."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= target].max())


def compute_metrics(y, p, thr: float = 0.5) -> dict:
    """Threshold metrics at `thr` plus ranking metrics, Brier score and recall at fixed FPR."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "threshold": float(thr),
        "precision": prec, "recall": rec,
        "f1": (2 * prec * rec / (prec + rec)) if prec + rec and not math.isnan(rec) else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "brier_score": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
    if len(np.unique(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                   recall_at_1pct_fpr=recall_at_fpr(y, p, 0.01), recall_at_5pct_fpr=recall_at_fpr(y, p, 0.05))
    else:
        out.update(roc_auc=float("nan"), pr_auc=float("nan"),
                   recall_at_1pct_fpr=float("nan"), recall_at_5pct_fpr=float("nan"))
    return out


def metrics_by_source(source, y, p, thr: float = 0.5) -> dict:
    """Recall for malicious sources, false-positive rate for benign sources."""
    source, y, p = np.asarray(source), np.asarray(y), np.asarray(p, dtype=float)
    res = {}
    for s in sorted(set(source)):
        m = source == s
        flagged = p[m] >= thr
        key = "recall" if y[m].mean() == 1 else "false_positive_rate"
        res[s] = {"n": int(m.sum()), key: float(flagged.mean()), "mean_score": float(p[m].mean())}
    return res


def build_report(source, y, p, vec, thr: float = 0.5) -> dict:
    """Full report: all rows, rows deduplicated by feature vector, and per-source numbers."""
    first = ~pd.Series(np.asarray(vec)).duplicated().to_numpy()
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return {
        "all": compute_metrics(y, p, thr),
        "dedup": compute_metrics(y[first], p[first], thr),
        "by_source": metrics_by_source(source, y, p, thr),
    }


def policy_outcomes(y, p, thresholds: dict) -> dict:
    """What each allow/warn/block policy does to a set of scores."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {}
    for name, t in thresholds.items():
        block = p >= t["block"]
        flagged = p >= t["warn"]
        out[name] = {
            "block_recall": float(block[y == 1].mean()), "block_fpr": float(block[y == 0].mean()),
            "warn_or_block_recall": float(flagged[y == 1].mean()),
            "warn_or_block_fpr": float(flagged[y == 0].mean()),
        }
    return out


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_curves(preds: dict, out_dir: Path) -> list[str]:
    """ROC and precision-recall curves, one line per model. preds = {name: (y, p)}."""
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p) in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (holdout)")
    ax.legend()
    paths.append(_save(fig, out_dir / "roc.png"))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p) in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (holdout)")
    ax.legend()
    paths.append(_save(fig, out_dir / "pr.png"))
    return paths


def plot_confusion(y, p, thr: float, path: Path, title: str = "Confusion matrix") -> str:
    m = compute_metrics(y, p, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["benign", "malicious"],
           yticklabels=["benign", "malicious"], xlabel="Predicted", ylabel="Actual", title=title)
    return _save(fig, path)


def plot_calibration(curves: dict, path: Path) -> str:
    """Reliability curves. curves = {name: (y, p)}."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p) in curves.items():
        frac, mean = calibration_curve(y, np.clip(p, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="Predicted risk", ylabel="Observed malicious rate", title="Calibration (holdout)")
    ax.legend()
    return _save(fig, path)


def plot_importance(names, mean, std, path: Path, top: int = 20) -> str:
    order = np.argsort(mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh([names[i] for i in order], np.asarray(mean)[order], xerr=np.asarray(std)[order])
    ax.set(xlabel="Drop in average precision when shuffled", title="Permutation importance (holdout)")
    return _save(fig, path)


HEADLINE_METRICS = ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
                    "recall_at_1pct_fpr", "recall_at_5pct_fpr", "brier_score")


def mlflow_metrics(report: dict, prefix: str) -> dict:
    """Flat, readable metric names for MLflow, e.g. holdout_pr_auc or holdout_recall_datadog."""
    out = {f"{prefix}_{name}": report["all"][name] for name in HEADLINE_METRICS}
    out[f"{prefix}_dedup_pr_auc"] = report["dedup"]["pr_auc"]
    out[f"{prefix}_dedup_recall"] = report["dedup"]["recall"]
    for source, values in report["by_source"].items():
        key = "recall" if "recall" in values else "false_positive_rate"
        out[f"{prefix}_{key}_{source}"] = values[key]
    return out
