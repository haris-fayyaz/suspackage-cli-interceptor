"""Holdout protocol: same-distribution test. Tune on dev folds, score once on the untouched holdout."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.inspection import permutation_importance

from .ablations import run_ablations
from .calibration import brier_summary, fit_calibrator
from .context import HoldoutData, TrainingContext, build_holdout_data
from .evaluate import (build_report, mlflow_metrics, plot_calibration, plot_confusion, plot_curves,
                       plot_importance, policy_outcomes)
from .fitting import pipeline_factory, predict_out_of_fold, save_json, tune_hyperparameters
from .models import DEFAULT_BUILDER
from .thresholds import choose_thresholds

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    hyperparameters: dict
    cv_scores: np.ndarray        # out-of-fold scores on the dev set
    holdout_scores: np.ndarray
    fitted_pipeline: object
    cv_report: dict
    holdout_report: dict


def train_and_evaluate_model(ctx: TrainingContext, data: HoldoutData, model_name: str) -> ModelResult:
    """Tune (if the model has a search space), then score with grouped CV and on the holdout."""
    with ctx.tracker.run(f"holdout-{model_name}", nested=True, tags={"model": model_name, "protocol": "holdout"}):
        if model_name in ctx.settings["search"]:
            hyperparameters = tune_hyperparameters(ctx, model_name, data.dev_inputs, data.dev_labels, data.folds)
        else:
            hyperparameters = dict(ctx.settings["defaults"].get(model_name, {}))
        make_pipeline = pipeline_factory(ctx, model_name, hyperparameters, DEFAULT_BUILDER, data.dev_labels)
        cv_scores = predict_out_of_fold(make_pipeline, data.dev_inputs, data.dev_labels, data.folds)
        rows = data.dev_training_rows
        pipeline = make_pipeline().fit(data.dev_inputs.iloc[rows], data.dev_labels[rows])
        holdout_scores = pipeline.predict_proba(data.holdout_inputs)[:, 1]
        cv_report = build_report(data.dev_sources, data.dev_labels, cv_scores, data.dev_vectors)
        holdout_report = build_report(data.holdout_sources, data.holdout_labels, holdout_scores,
                                      data.holdout_vectors)
        ctx.tracker.params({"model": model_name, **hyperparameters})
        ctx.tracker.metrics({**mlflow_metrics(cv_report, "cv"), **mlflow_metrics(holdout_report, "holdout")})
    return ModelResult(model_name, hyperparameters, cv_scores, holdout_scores, pipeline, cv_report, holdout_report)


def pick_best_model(results: dict[str, ModelResult]) -> str:
    """Best non-baseline model by cross-validated PR-AUC. The holdout is never used to choose."""
    candidates = [name for name in results if name != "dummy_baseline"]
    return max(candidates, key=lambda name: results[name].cv_report["all"]["pr_auc"])


def compute_permutation_importance(ctx: TrainingContext, data: HoldoutData, best: ModelResult) -> dict:
    """How much average precision drops on the holdout when one input column is shuffled."""
    result = permutation_importance(
        best.fitted_pipeline, data.holdout_inputs, data.holdout_labels, scoring="average_precision",
        n_repeats=int(ctx.settings["permutation_repeats"]), random_state=ctx.seed, n_jobs=1)
    return {column: {"mean": float(mean), "std": float(std)} for column, mean, std
            in zip(ctx.input_columns, result.importances_mean, result.importances_std)}


def calibrate_and_choose_thresholds(ctx: TrainingContext, data: HoldoutData, best: ModelResult) -> dict:
    """Calibrate on dev out-of-fold scores, choose thresholds there, then check them on the holdout."""
    calibrator = fit_calibrator(best.cv_scores, data.dev_labels)
    calibrated_dev = calibrator.predict(best.cv_scores)
    calibrated_holdout = calibrator.predict(best.holdout_scores)
    thresholds = choose_thresholds(data.dev_labels, calibrated_dev, ctx.settings["policies"])
    return {
        "calibrated_holdout_scores": calibrated_holdout,
        "thresholds": thresholds,
        "brier": {**brier_summary(data.dev_labels, best.cv_scores, calibrated_dev, "dev"),
                  **brier_summary(data.holdout_labels, best.holdout_scores, calibrated_holdout, "holdout")},
        "policies_on_holdout": policy_outcomes(data.holdout_labels, calibrated_holdout, thresholds),
    }


def save_figures(ctx: TrainingContext, data: HoldoutData, results: dict, best: ModelResult,
                 calibration: dict, importance: dict) -> list[str]:
    figures = ctx.reports_dir / "figures"
    labels = data.holdout_labels
    paths = plot_curves({name: (labels, r.holdout_scores) for name, r in results.items()}, figures)
    block = calibration["thresholds"]["balanced"]["block"]
    paths.append(plot_confusion(labels, calibration["calibrated_holdout_scores"], block,
                                figures / "confusion_matrix.png",
                                f"{best.name}: block decisions, balanced policy (holdout)"))
    paths.append(plot_calibration({"raw": (labels, best.holdout_scores),
                                   "calibrated": (labels, calibration["calibrated_holdout_scores"])},
                                  figures / "calibration.png"))
    ordered = [importance[c] for c in ctx.input_columns]
    paths.append(plot_importance(ctx.input_columns, [o["mean"] for o in ordered],
                                 [o["std"] for o in ordered], figures / "importance.png"))
    return paths


def run_holdout_protocol(ctx: TrainingContext) -> dict:
    """Run all models, the ablations, calibration and thresholds. Returns the holdout report."""
    data = build_holdout_data(ctx)
    results = {name: train_and_evaluate_model(ctx, data, name) for name in ctx.settings["models"]}
    best_name = pick_best_model(results)
    best = results[best_name]
    log.info("best model by cross-validated PR-AUC: %s", best_name)

    ablations = run_ablations(ctx, data, best_name, best.hyperparameters)
    use_density = (ablations["full"]["cv"]["all"]["pr_auc"]
                   >= ablations["raw_counts_no_density"]["cv"]["all"]["pr_auc"])
    importance = compute_permutation_importance(ctx, data, best)
    calibration = calibrate_and_choose_thresholds(ctx, data, best)
    figure_paths = save_figures(ctx, data, results, best, calibration, importance)
    ctx.tracker.metrics(calibration["brier"])
    for path in figure_paths:
        ctx.tracker.artifact(path, "figures")

    save_json(ctx.reports_dir / "best_params.json", {
        "model": best_name, "hyperparameters": best.hyperparameters, "use_density": bool(use_density),
        "hyperparameters_by_model": {name: r.hyperparameters for name, r in results.items()},
        "search_iterations": int(ctx.settings["search_iter"]),
        "tuned_on": "holdout protocol dev folds, scored by average precision"})
    return {
        "split": {"dev_rows": int(len(data.dev_labels)), "holdout_rows": int(len(data.holdout_labels)),
                  "holdout_malicious": int(data.holdout_labels.sum()),
                  "capped_dev_training_rows": int(len(data.dev_training_rows))},
        "models": {name: {"hyperparameters": r.hyperparameters, "cv": r.cv_report, "holdout": r.holdout_report}
                   for name, r in results.items()},
        "best_model": best_name, "ablations": ablations, "permutation_importance": importance,
        "calibration": calibration["brier"], "thresholds_from_dev_out_of_fold": calibration["thresholds"],
        "policies_on_holdout": calibration["policies_on_holdout"],
    }
