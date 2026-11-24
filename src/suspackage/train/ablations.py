"""Ablations: refit the best model with feature groups removed, to test for shortcut learning."""
from __future__ import annotations

import logging

from .context import HoldoutData, TrainingContext
from .evaluate import build_report, mlflow_metrics
from .fitting import pipeline_factory, predict_out_of_fold
from .models import DEFAULT_BUILDER, feature_names

log = logging.getLogger(__name__)

# variant name -> feature options for the pipeline builder
ABLATION_VARIANTS = {
    "full": {},
    "no_name_features": {"drop_prefixes": ["name_"], "drop_cols": ["is_typosquat"]},
    "no_metadata_features": {"drop_prefixes": ["meta_"]},
    "no_sp_present_and_metadata": {"drop_prefixes": ["meta_"], "drop_cols": ["sp_present"]},
    "raw_counts_no_density": {"use_density": False},
}


def run_ablations(ctx: TrainingContext, data: HoldoutData, model_name: str, hyperparameters: dict) -> dict:
    """One MLflow run per variant. Each is scored with grouped CV on dev and on the holdout."""
    results = {}
    for variant, options in ABLATION_VARIANTS.items():
        feature_options = {**DEFAULT_BUILDER, **options}
        make_pipeline = pipeline_factory(ctx, model_name, hyperparameters, feature_options, data.dev_labels)
        tags = {"model": model_name, "protocol": "ablation", "ablation": variant}
        with ctx.tracker.run(f"ablation-{variant}", nested=True, tags=tags):
            cv_scores = predict_out_of_fold(make_pipeline, data.dev_inputs, data.dev_labels, data.folds)
            rows = data.dev_training_rows
            pipeline = make_pipeline().fit(data.dev_inputs.iloc[rows], data.dev_labels[rows])
            holdout_scores = pipeline.predict_proba(data.holdout_inputs)[:, 1]
            cv_report = build_report(data.dev_sources, data.dev_labels, cv_scores, data.dev_vectors)
            holdout_report = build_report(data.holdout_sources, data.holdout_labels, holdout_scores,
                                          data.holdout_vectors)
            ctx.tracker.params({"model": model_name, "ablation": variant, **hyperparameters})
            ctx.tracker.metrics({**mlflow_metrics(cv_report, "cv"), **mlflow_metrics(holdout_report, "holdout")})
        results[variant] = {"cv": cv_report, "holdout": holdout_report,
                            "n_model_inputs": len(feature_names(pipeline))}
        log.info("ablation %-28s holdout PR-AUC %.3f", variant, holdout_report["all"]["pr_auc"])
    return results
