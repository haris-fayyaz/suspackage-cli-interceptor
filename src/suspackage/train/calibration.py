"""Turn raw model scores into calibrated risk scores in [0, 1]."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_calibrator(out_of_fold_scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Isotonic calibration fitted on out-of-fold scores, so it never sees training-set scores."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(out_of_fold_scores, labels)


def brier_summary(labels: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, prefix: str) -> dict:
    """Brier score before and after calibration. Lower is better."""
    return {f"{prefix}_brier_raw": float(brier_score_loss(labels, raw)),
            f"{prefix}_brier_calibrated": float(brier_score_loss(labels, calibrated))}
