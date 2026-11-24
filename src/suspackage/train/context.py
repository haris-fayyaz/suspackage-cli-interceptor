"""Everything a training step needs, loaded once: the data, labels, split groups and tracker."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .splits import limit_rows_per_group, make_grouped_folds, make_holdout_split
from .tracking import Tracker, git_commit

log = logging.getLogger(__name__)


@dataclass
class TrainingContext:
    settings: dict              # the `train:` section of params.yaml
    rows: pd.DataFrame          # usable rows (ok == 1), one per package version
    inputs: pd.DataFrame        # model input columns for every row
    labels: np.ndarray          # 1 = malicious, 0 = benign
    groups: np.ndarray          # split group id per row (families and identical vectors merged)
    vectors: np.ndarray         # hash of each row's feature vector
    input_columns: list[str]
    tracker: Tracker
    reports_dir: Path
    models_dir: Path
    data_version: str

    @property
    def seed(self) -> int:
        return int(self.settings["seed"])

    @property
    def sources(self) -> np.ndarray:
        return self.rows["source"].to_numpy()


def build_training_context(params: dict, tracker: Tracker, features_path: str | None = None,
                           dvc_file=("data/features.dvc", "features.dvc")) -> TrainingContext:
    """Load features.csv, keep usable rows and build the split groups."""
    settings = params["train"]
    path = features_path or params["paths"]["features"]
    rows = data.usable_rows(data.load_features(path))
    feature_cols = data.feature_columns(rows)
    columns = data.input_columns(rows)
    groups = data.build_split_groups(rows, feature_cols)
    stats = data.group_stats(groups)
    log.info("split groups: %s", stats)
    if stats["share_largest"] > 0.05:
        log.warning("one split group holds %.1f%% of all rows. Splits and the per-group cap may be distorted.",
                    100 * stats["share_largest"])
    return TrainingContext(
        settings=settings, rows=rows, inputs=rows[columns].astype(float),
        labels=rows["label"].to_numpy().astype(int), groups=groups,
        vectors=data.feature_vector_hash(rows, feature_cols), input_columns=columns, tracker=tracker,
        reports_dir=Path(settings.get("reports_dir", "reports")),
        models_dir=Path(settings.get("models_dir", "models")),
        data_version=data.data_version(dvc_file, path),
    )


def data_summary(ctx: TrainingContext) -> dict:
    """Row counts, family and group sizes. Goes into metrics.json under `data`."""
    return {
        "rows": int(len(ctx.rows)), "malicious": int(ctx.labels.sum()),
        "n_features": len(data.feature_columns(ctx.rows)),
        "families": data.family_stats(ctx.rows), "split_groups": data.group_stats(ctx.groups),
        "rows_with_duplicate_feature_vector": int(pd.Series(ctx.vectors).duplicated().sum()),
        "data_version": ctx.data_version, "git_commit": git_commit(), "seed": ctx.seed,
    }


@dataclass
class HoldoutData:
    """Dev and holdout parts of the data for the holdout protocol. Never mix them up."""
    dev_inputs: pd.DataFrame
    dev_labels: np.ndarray
    dev_sources: np.ndarray
    dev_vectors: np.ndarray
    folds: list                 # grouped CV folds inside the dev set
    dev_training_rows: np.ndarray   # dev rows used to fit the holdout models (capped per group)
    holdout_inputs: pd.DataFrame
    holdout_labels: np.ndarray
    holdout_sources: np.ndarray
    holdout_vectors: np.ndarray


def build_holdout_data(ctx: TrainingContext) -> HoldoutData:
    """Group holdout (never tuned on) plus 5 grouped folds inside the dev part."""
    s = ctx.settings
    dev, hold = make_holdout_split(ctx.labels, ctx.groups, s["holdout_size"], ctx.seed)
    dev_groups, dev_labels = ctx.groups[dev], ctx.labels[dev]
    folds = make_grouped_folds(dev_labels, dev_groups, s["n_folds"], ctx.seed, s["group_cap"])
    training_rows = limit_rows_per_group(np.arange(len(dev)), dev_groups, s["group_cap"], ctx.seed)
    log.info("holdout split: dev %d rows, holdout %d rows, capped dev training rows %d",
             len(dev), len(hold), len(training_rows))
    return HoldoutData(
        dev_inputs=ctx.inputs.iloc[dev], dev_labels=dev_labels, dev_sources=ctx.sources[dev],
        dev_vectors=ctx.vectors[dev], folds=folds, dev_training_rows=training_rows,
        holdout_inputs=ctx.inputs.iloc[hold], holdout_labels=ctx.labels[hold],
        holdout_sources=ctx.sources[hold], holdout_vectors=ctx.vectors[hold],
    )
