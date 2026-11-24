"""Cross-dataset protocol: train on pypi_malregistry malware, test on DataDog malware."""
from __future__ import annotations

import logging

from .context import TrainingContext
from .evaluate import build_report, mlflow_metrics
from .fitting import load_best_settings, pipeline_factory
from .models import DEFAULT_BUILDER
from .splits import limit_rows_per_group, make_cross_dataset_split

log = logging.getLogger(__name__)


def run_cross_dataset_protocol(ctx: TrainingContext) -> dict:
    """Does the model catch malware from a source it never saw? The dummy baseline is skipped."""
    best = load_best_settings(ctx.reports_dir)
    split = make_cross_dataset_split(ctx.sources, ctx.labels, ctx.groups, ctx.seed,
                                     ctx.settings["cross_benign_test_frac"])
    train_rows = limit_rows_per_group(split.train, ctx.groups, ctx.settings["group_cap"], ctx.seed)
    test_rows = split.test
    feature_options = {**DEFAULT_BUILDER, "use_density": best["use_density"]}
    report = {"split": {**split.stats, "capped_train_rows": int(len(train_rows))}, "models": {}}

    for model_name in (m for m in ctx.settings["models"] if m != "dummy_baseline"):
        hyperparameters = best["hyperparameters_by_model"].get(model_name, {})
        make_pipeline = pipeline_factory(ctx, model_name, hyperparameters, feature_options, ctx.labels[train_rows])
        with ctx.tracker.run(f"cross-{model_name}", nested=True,
                             tags={"model": model_name, "protocol": "cross_dataset"}):
            pipeline = make_pipeline().fit(ctx.inputs.iloc[train_rows], ctx.labels[train_rows])
            scores = pipeline.predict_proba(ctx.inputs.iloc[test_rows])[:, 1]
            model_report = build_report(ctx.sources[test_rows], ctx.labels[test_rows], scores,
                                        ctx.vectors[test_rows])
            ctx.tracker.params({"model": model_name, **hyperparameters})
            ctx.tracker.metrics(mlflow_metrics(model_report, "cross_dataset"))
        report["models"][model_name] = model_report
        log.info("cross-dataset %-20s PR-AUC %.3f", model_name, model_report["all"]["pr_auc"])
    return report
