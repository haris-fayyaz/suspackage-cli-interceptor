"""Entry point:  python -m suspackage.train --protocol holdout|cross_dataset|final|all"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import load_params
from .context import build_training_context, data_summary
from .cross_dataset import run_cross_dataset_protocol
from .evaluate import clean
from .final_model import fit_final_model
from .holdout import run_holdout_protocol
from .tracking import Tracker, git_commit

log = logging.getLogger("suspackage.train")
PARENT_RUN_NAME = "training"


def update_metrics_file(path: Path, section: str, content: dict) -> dict:
    """Write one section of reports/metrics.json, keeping the sections written by earlier runs."""
    current = json.loads(path.read_text()) if path.exists() else {}
    current[section] = clean(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m suspackage.train")
    parser.add_argument("--protocol", choices=["holdout", "cross_dataset", "final", "all"], required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--features", default=None, help="override the path to features.csv")
    parser.add_argument("--search-iter", type=int, default=None,
                        help="override train.search_iter (use 2 for a smoke run)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    params = load_params(args.params)
    if args.search_iter:
        params["train"]["search_iter"] = args.search_iter
    tracker = Tracker(params["train"]["mlflow"], enabled=not args.no_mlflow)
    ctx = build_training_context(params, tracker, args.features)
    metrics_path = ctx.reports_dir / "metrics.json"
    steps = ["holdout", "cross_dataset", "final"] if args.protocol == "all" else [args.protocol]

    parent_tags = {"git_commit": git_commit(), "data_version": ctx.data_version, "protocols": ",".join(steps)}
    with tracker.run(PARENT_RUN_NAME, tags=parent_tags):
        tracker.params({k: params["train"][k] for k in
                        ("seed", "n_folds", "holdout_size", "group_cap", "search_iter", "cross_benign_test_frac")})
        metrics = update_metrics_file(metrics_path, "data", data_summary(ctx))
        if "holdout" in steps:
            metrics = update_metrics_file(metrics_path, "holdout", run_holdout_protocol(ctx))
        if "cross_dataset" in steps:
            metrics = update_metrics_file(metrics_path, "cross_dataset", run_cross_dataset_protocol(ctx))
        if "final" in steps:
            metrics = update_metrics_file(metrics_path, "final", fit_final_model(ctx, metrics))
        tracker.artifact(metrics_path, "reports")
        tracker.artifact(ctx.reports_dir / "best_params.json", "reports")
    log.info("done. metrics: %s | MLflow mode: %s", metrics_path, tracker.mode)
    return 0
