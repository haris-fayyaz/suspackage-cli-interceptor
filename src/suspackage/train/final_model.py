"""Fit the shipped model on all usable data, choose thresholds, save and register the bundle."""
from __future__ import annotations

import logging

import numpy as np

from ..model_bundle import ModelBundle
from .calibration import brier_summary, fit_calibrator
from .card import write_model_card
from .context import TrainingContext
from .evaluate import build_report, clean, mlflow_metrics
from .fitting import load_best_settings, pipeline_factory, predict_out_of_fold, save_json
from .splits import limit_rows_per_group, make_grouped_folds
from .thresholds import choose_thresholds
from .tracking import git_commit

log = logging.getLogger(__name__)
REGISTERED_MODEL_NAME = "suspackage-risk-model"


def fit_final_model(ctx: TrainingContext, metrics_so_far: dict) -> dict:
    """Out-of-fold scores on all data give the calibrator and thresholds. Then fit on everything."""
    s = ctx.settings
    best = load_best_settings(ctx.reports_dir)
    model_name, hyperparameters = best["model"], best["hyperparameters"]
    feature_options = {"drop_prefixes": list(s.get("final", {}).get("drop_prefixes", [])),
                       "drop_cols": list(s.get("final", {}).get("drop_cols", [])),
                       "use_density": bool(best["use_density"])}
    make_pipeline = pipeline_factory(ctx, model_name, hyperparameters, feature_options, ctx.labels)

    folds = make_grouped_folds(ctx.labels, ctx.groups, s["n_folds"], ctx.seed, s["group_cap"])
    raw_scores = predict_out_of_fold(make_pipeline, ctx.inputs, ctx.labels, folds)
    calibrator = fit_calibrator(raw_scores, ctx.labels)
    calibrated_scores = calibrator.predict(raw_scores)
    thresholds = choose_thresholds(ctx.labels, calibrated_scores, s["policies"])

    training_rows = limit_rows_per_group(np.arange(len(ctx.labels)), ctx.groups, s["group_cap"], ctx.seed)
    pipeline = make_pipeline().fit(ctx.inputs.iloc[training_rows], ctx.labels[training_rows])
    bundle = ModelBundle(
        pipeline=pipeline, feature_columns=list(ctx.input_columns), thresholds=clean(thresholds),
        data_hash=ctx.data_version, calibrator=calibrator, model_name=model_name,
        meta={"hyperparameters": clean(hyperparameters), "feature_options": feature_options,
              "training_rows": int(len(training_rows)), "seed": ctx.seed, "git_commit": git_commit()})
    model_path = ctx.models_dir / "suspackage_model.joblib"
    bundle.save(model_path)
    _check_saved_bundle_matches(ctx, model_path, pipeline, calibrator)

    save_json(ctx.models_dir / "feature_columns.json", {
        "feature_columns": ctx.input_columns, "schema_version": bundle.schema_version,
        "note": "x_py_kb is used only to compute per-KB densities"})
    save_json(ctx.models_dir / "thresholds.json", {
        "policies": thresholds, "score": "calibrated risk in [0, 1]",
        "rule": "score >= block blocks, score >= warn warns, otherwise allow"})

    out_of_fold_report = build_report(ctx.sources, ctx.labels, calibrated_scores, ctx.vectors)
    write_model_card(ctx.models_dir / "model_card.md",
                     {"model": model_name, "n_train": len(training_rows), "data_hash": ctx.data_version,
                      "thresholds": thresholds, "columns": ctx.input_columns, "oof": out_of_fold_report["all"]},
                     metrics_so_far)
    _log_to_mlflow(ctx, model_path, out_of_fold_report, thresholds)
    return {"model": model_name, "hyperparameters": hyperparameters, "feature_options": feature_options,
            "training_rows": int(len(training_rows)), "out_of_fold_calibrated": out_of_fold_report,
            "thresholds": thresholds,
            "calibration": brier_summary(ctx.labels, raw_scores, calibrated_scores, "out_of_fold")}


def _check_saved_bundle_matches(ctx: TrainingContext, model_path, pipeline, calibrator) -> None:
    """The file on disk must give the same scores as the model in memory."""
    loaded = ModelBundle.load(model_path)
    expected = calibrator.predict(pipeline.predict_proba(ctx.inputs.iloc[:200])[:, 1])
    if not np.allclose(loaded.predict_frame(ctx.rows.iloc[:200]), expected):
        raise RuntimeError("saved bundle does not reproduce the in-memory predictions")


def _log_to_mlflow(ctx: TrainingContext, model_path, report: dict, thresholds: dict) -> None:
    """Log final metrics and files to the active (parent) run and register the model."""
    threshold_values = {f"threshold_{policy}_{level}": values[level]
                        for policy, values in thresholds.items() for level in ("warn", "block")}
    ctx.tracker.metrics({**mlflow_metrics(report, "final_out_of_fold"), **threshold_values})
    ctx.tracker.dict_artifact(clean(thresholds), "thresholds.json")
    for name in ("feature_columns.json", "thresholds.json", "model_card.md"):
        ctx.tracker.artifact(ctx.models_dir / name, "model_files")
    ctx.tracker.log_model_bundle(model_path, REGISTERED_MODEL_NAME)
