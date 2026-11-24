"""Shared fitting helpers: build pipelines, out-of-fold predictions, hyperparameter search."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .context import TrainingContext
from .evaluate import clean
from .models import DEFAULT_BUILDER, build_pipeline, positive_weight

log = logging.getLogger(__name__)
PipelineFactory = Callable[[], Pipeline]


def pipeline_factory(ctx: TrainingContext, model_name: str, hyperparameters: dict,
                     feature_options: dict, training_labels: np.ndarray) -> PipelineFactory:
    """Return a function that builds a fresh, unfitted pipeline each time it is called."""
    weight = positive_weight(training_labels)
    return lambda: build_pipeline(model_name, ctx.input_columns, ctx.seed, hyperparameters,
                                  feature_options, weight)


def predict_out_of_fold(make_pipeline: PipelineFactory, inputs: pd.DataFrame, labels: np.ndarray,
                        folds: list) -> np.ndarray:
    """Score every row with a model that never saw it (or its group) in training."""
    scores = np.full(len(labels), np.nan)
    for train_rows, validation_rows in folds:
        pipeline = make_pipeline()
        pipeline.fit(inputs.iloc[train_rows], labels[train_rows])
        scores[validation_rows] = pipeline.predict_proba(inputs.iloc[validation_rows])[:, 1]
    if np.isnan(scores).any():
        raise RuntimeError("some rows were never in a validation fold")
    return scores


def tune_hyperparameters(ctx: TrainingContext, model_name: str, inputs: pd.DataFrame,
                         labels: np.ndarray, folds: list) -> dict:
    """Small random search scored by average precision on grouped folds. Test rows are never used."""
    search_space = ctx.settings["search"][model_name]
    n_combinations = int(np.prod([len(v) for v in search_space.values()]))
    search = RandomizedSearchCV(
        pipeline_factory(ctx, model_name, ctx.settings["defaults"].get(model_name, {}),
                         DEFAULT_BUILDER, labels)(),
        {f"clf__{name}": values for name, values in search_space.items()},
        n_iter=min(int(ctx.settings["search_iter"]), n_combinations), scoring="average_precision",
        cv=folds, refit=False, random_state=ctx.seed, n_jobs=1, error_score="raise")
    search.fit(inputs, labels)
    scores = search.cv_results_["mean_test_score"]
    best = search.cv_results_["params"][int(np.argmax(scores))]
    hyperparameters = {name.removeprefix("clf__"): value for name, value in best.items()}
    log.info("%s best hyperparameters (cv average precision %.4f): %s", model_name, scores.max(), hyperparameters)
    return hyperparameters


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(content), indent=2), encoding="utf-8")


def load_best_settings(reports_dir: Path) -> dict:
    """Best model and hyperparameters written by the holdout protocol."""
    path = reports_dir / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the holdout protocol first.")
    return json.loads(path.read_text())
