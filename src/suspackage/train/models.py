"""Model pipelines. Every pipeline takes the raw input columns (features + x_py_kb)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import COUNT_COLS, DENSITY_BASE, DENSITY_COLS

MODEL_NAMES = ("dummy_baseline", "logistic_regression", "random_forest", "xgboost")
DEFAULT_BUILDER = {"drop_prefixes": [], "drop_cols": [], "use_density": True}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Select columns, add per-KB densities, optionally log1p counts.

    Densities are count / (x_py_kb + 1). x_py_kb itself is never passed to the model.
    """

    def __init__(self, columns, drop_prefixes=(), drop_cols=(), use_density=True, log1p=False):
        self.columns = columns
        self.drop_prefixes = drop_prefixes
        self.drop_cols = drop_cols
        self.use_density = use_density
        self.log1p = log1p

    def _frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=list(self.columns))
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"missing input columns: {missing}")
        return X[list(self.columns)]

    def fit(self, X, y=None):
        self._frame(X)
        drop, prefixes = set(self.drop_cols), tuple(self.drop_prefixes)
        kept = [c for c in self.columns
                if c != DENSITY_BASE and c not in drop and not c.startswith(prefixes)]
        dens = [c for c in DENSITY_COLS if c in kept] if (self.use_density and DENSITY_BASE in self.columns) else []
        self.kept_ = kept
        self.density_cols_ = dens
        self.feature_names_ = kept + [f"d_{c}" for c in dens]
        if self.log1p:
            self.log_idx_ = [i for i, n in enumerate(self.feature_names_) if n in COUNT_COLS or n.startswith("d_")]
        else:
            self.log_idx_ = []
        return self

    def transform(self, X):
        X = self._frame(X)
        out = X[self.kept_].to_numpy(dtype=float)
        if self.density_cols_:
            kb = X[DENSITY_BASE].to_numpy(dtype=float) + 1.0
            dens = X[self.density_cols_].to_numpy(dtype=float) / kb[:, None]
            out = np.hstack([out, dens])
        if self.log_idx_:
            out = out.copy()
            out[:, self.log_idx_] = np.log1p(np.clip(out[:, self.log_idx_], 0, None))
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def build_pipeline(model_name: str, columns: list[str], seed: int, model_params: dict | None = None,
                   builder: dict | None = None, scale_pos_weight: float = 1.0) -> Pipeline:
    """Build an unfitted pipeline. `model_params` are classifier params without a prefix."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model_name}")
    b = {**DEFAULT_BUILDER, **(builder or {})}
    p = dict(model_params or {})
    feats = FeatureBuilder(columns, list(b["drop_prefixes"]), list(b["drop_cols"]),
                           b["use_density"], log1p=(model_name == "logistic_regression"))
    if model_name == "dummy_baseline":
        return Pipeline([("features", feats), ("clf", DummyClassifier(strategy="prior"))])
    if model_name == "logistic_regression":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed, **p)
        return Pipeline([("features", feats), ("scale", StandardScaler()), ("clf", clf)])
    if model_name == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **p)
        return Pipeline([("features", feats), ("clf", clf)])
    clf = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed,
                        scale_pos_weight=scale_pos_weight, **p)
    return Pipeline([("features", feats), ("clf", clf)])


def feature_names(pipe: Pipeline) -> list[str]:
    """Names of the columns the classifier sees (after the builder)."""
    return list(pipe.named_steps["features"].feature_names_)


def positive_weight(y: np.ndarray) -> float:
    """negatives / positives, for XGBoost class balance."""
    pos = max(int((np.asarray(y) == 1).sum()), 1)
    return float((np.asarray(y) == 0).sum() / pos)
