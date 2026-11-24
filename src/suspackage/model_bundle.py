"""The trained model plus everything the CLI needs to use it.

CLI contract:  archive -> featurize() (extract.py) -> ModelBundle.predict_proba -> decide

    bundle = ModelBundle.load("models/suspackage_model.joblib")
    score = bundle.predict_proba(features)      # features = dict from featurize()
    action = bundle.decide(score, "balanced")  # "allow" | "warn" | "block"

Only load bundles built by this project: joblib files are pickles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .train.thresholds import decide as _decide

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1
POLICIES = ("lenient", "balanced", "strict")


@dataclass
class ModelBundle:
    pipeline: Any                      # fitted sklearn Pipeline, takes a DataFrame of feature_columns
    feature_columns: list[str]         # ordered inputs; includes x_py_kb (used only for densities)
    thresholds: dict                   # {policy: {"warn": float, "block": float, ...}}
    data_hash: str = "unknown"         # md5 of the tracked training data
    schema_version: int = SCHEMA_VERSION
    calibrator: Any = None             # optional isotonic regression: raw score -> risk
    model_name: str = ""
    meta: dict = field(default_factory=dict)

    def _frame(self, rows: list[list[float]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=self.feature_columns, dtype=float)

    def _score(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(frame)[:, 1]
        out = self.calibrator.predict(raw) if self.calibrator is not None else raw
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Risk score in [0, 1]. Missing features default to 0 (with a warning), extra keys are ignored."""
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            log.warning("missing features defaulted to 0: %s", ", ".join(missing))
        row = [float(features.get(c, 0.0)) for c in self.feature_columns]
        return float(self._score(self._frame([row]))[0])

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Batch scoring for a DataFrame that has the feature columns."""
        return self._score(df[self.feature_columns].astype(float))

    def decide(self, score: float, policy: str = "balanced") -> str:
        """allow, warn or block for one policy."""
        if policy not in self.thresholds:
            raise ValueError(f"unknown policy {policy!r}, choose from {sorted(self.thresholds)}")
        return _decide(score, self.thresholds[policy])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} is not a ModelBundle")
        if bundle.schema_version != SCHEMA_VERSION:
            raise ValueError(f"bundle schema {bundle.schema_version}, code expects {SCHEMA_VERSION}")
        return bundle
