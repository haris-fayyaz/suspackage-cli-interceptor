"""Runs every protocol once on a small synthetic table with a local MLflow store."""
import json
from pathlib import Path

import mlflow
import pytest
import yaml
from conftest import make_features

from suspackage.model_bundle import ModelBundle
from suspackage.train.cli import main

MODELS = ("dummy_baseline", "logistic_regression", "random_forest", "xgboost")
ABLATIONS = ("full", "no_name_features", "no_metadata_features", "no_sp_present_and_metadata",
             "raw_counts_no_density")


def write_workspace(work: Path) -> None:
    """params.yaml with tiny search spaces plus a synthetic features.csv."""
    base = yaml.safe_load((Path(__file__).parents[1] / "params.yaml").read_text())
    train = base["train"]
    train.update(search_iter=2, n_folds=3, permutation_repeats=2, group_cap=5)
    train["search"] = {"random_forest": {"n_estimators": [20, 30], "max_depth": [4, 8]},
                       "xgboost": {"n_estimators": [20, 30], "max_depth": [3, 4]}}
    train["defaults"] = {"logistic_regression": {"C": 1.0}, "random_forest": {"n_estimators": 20},
                         "xgboost": {"n_estimators": 20}}
    train["mlflow"]["tracking_uri"] = ""
    (work / "params.yaml").write_text(yaml.safe_dump({"paths": base["paths"], "train": train}))
    make_features(120).to_csv(work / "features.csv", index=False)


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    work = tmp_path_factory.mktemp("training")
    write_workspace(work)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        mp.setenv("MLFLOW_TRACKING_URI", "")
        assert main(["--protocol", "all", "--features", "features.csv"]) == 0
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        yield work


def test_artifacts_and_metrics_are_written(trained_dir):
    for f in ("models/suspackage_model.joblib", "models/feature_columns.json", "models/thresholds.json",
              "models/model_card.md", "reports/metrics.json", "reports/best_params.json",
              "reports/figures/roc.png", "reports/figures/pr.png", "reports/figures/confusion_matrix.png",
              "reports/figures/calibration.png", "reports/figures/importance.png"):
        assert (trained_dir / f).exists(), f
    metrics = json.loads((trained_dir / "reports/metrics.json").read_text())
    assert set(metrics) >= {"data", "holdout", "cross_dataset", "final"}
    holdout = metrics["holdout"]
    assert set(holdout["ablations"]) == set(ABLATIONS)
    assert holdout["best_model"] in MODELS[1:]
    assert "datadog" in metrics["cross_dataset"]["models"]["random_forest"]["by_source"]
    assert "dummy_baseline" not in metrics["cross_dataset"]["models"]
    bundle = ModelBundle.load(trained_dir / "models/suspackage_model.joblib")
    assert 0 <= bundle.predict_proba({}) <= 1


def test_mlflow_run_names_are_readable(trained_dir):
    """Teammates read these names in MLflow, so lock them down."""
    runs = mlflow.search_runs(experiment_names=["suspackage-training"])
    expected = {"training"}
    expected |= {f"holdout-{m}" for m in MODELS}
    expected |= {f"cross-{m}" for m in MODELS[1:]}
    expected |= {f"ablation-{v}" for v in ABLATIONS}
    assert set(runs["tags.mlflow.runName"]) == expected and len(runs) == 13
    parent = runs[runs["tags.mlflow.runName"] == "training"].iloc[0]
    assert parent["metrics.final_out_of_fold_pr_auc"] > 0
    child = runs[runs["tags.mlflow.runName"] == "holdout-random_forest"].iloc[0]
    assert child["tags.model"] == "random_forest" and child["metrics.holdout_pr_auc"] > 0


def test_model_is_registered_with_a_clear_name(trained_dir):
    client = mlflow.MlflowClient()
    assert [m.name for m in client.search_registered_models()] == ["suspackage-risk-model"]
    assert len(client.search_model_versions("name='suspackage-risk-model'")) == 1


def test_runs_without_mlflow(tmp_path, monkeypatch):
    """--no-mlflow must work end to end (it crashed once on Tracker.params)."""
    write_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["--protocol", "all", "--features", "features.csv", "--no-mlflow"]) == 0
    assert (tmp_path / "models/suspackage_model.joblib").exists()
    assert not (tmp_path / "mlruns").exists()
