"""Policy thresholds for allow / warn / block, chosen from out-of-fold scores."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def threshold_at_fpr(y, p, target: float) -> float:
    """Lowest threshold whose false-positive rate is at most `target`. Capped at 1.0."""
    fpr, _, thr = roc_curve(np.asarray(y), np.asarray(p, dtype=float))
    ok = np.flatnonzero(fpr <= target)
    t = float(thr[ok.max()]) if len(ok) else 1.0
    return min(t, 1.0)


def choose_thresholds(y, p, policies: dict) -> dict:
    """policies = {name: {block_fpr, warn_fpr}}. A score >= block blocks, >= warn warns.

    Stricter policies allow a higher FPR, so their thresholds are lower. warn <= block always.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {}
    for name, cfg in policies.items():
        block = threshold_at_fpr(y, p, cfg["block_fpr"])
        warn = min(threshold_at_fpr(y, p, cfg["warn_fpr"]), block)
        out[name] = {
            "warn": warn, "block": block,
            "block_fpr_target": cfg["block_fpr"], "warn_fpr_target": cfg["warn_fpr"],
            "oof_block_fpr": float((p[y == 0] >= block).mean()),
            "oof_block_recall": float((p[y == 1] >= block).mean()),
            "oof_warn_fpr": float((p[y == 0] >= warn).mean()),
            "oof_warn_recall": float((p[y == 1] >= warn).mean()),
        }
    return out


def decide(score: float, thr: dict) -> str:
    """Map a risk score to allow, warn or block using one policy's thresholds."""
    if score >= thr["block"]:
        return "block"
    if score >= thr["warn"]:
        return "warn"
    return "allow"
