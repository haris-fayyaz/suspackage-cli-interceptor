"""MLflow helpers. Falls back to a local ./mlruns folder (SQLite) when the server is unreachable.

Credentials come from MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD, never from code.
Logging never stops training: a failed log call prints a warning and training carries on.
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # git missing or not a repo
        return "unknown"


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict of numbers to {"a.b.c": value}. Skips NaN, None and non-numbers."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v)):
            out[key] = float(v)
    return out


class Tracker:
    """Thin MLflow wrapper. `enabled=False` makes every call a no-op."""

    def __init__(self, cfg: dict, enabled: bool = True, local_dir: str = "mlruns"):
        self.enabled = enabled
        self.mode = "off"
        self.cfg = cfg
        self.local_dir = local_dir
        if not enabled:
            return
        import mlflow

        self.mlflow = mlflow
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("tracking_uri")
        self.mode = "local"
        if uri and uri.startswith("http"):
            try:
                mlflow.set_tracking_uri(uri)
                mlflow.MlflowClient().search_experiments(max_results=1)
                mlflow.set_experiment(cfg["experiment"])
                self.mode = "remote"
            except Exception as e:
                log.warning("MLflow server unreachable (%s: %s). Using local ./%s instead.",
                            type(e).__name__, str(e)[:120], local_dir)
        if self.mode == "local":
            self._use_local()
        log.info("MLflow mode: %s", self.mode)

    def _use_local(self) -> None:
        """Local fallback: a SQLite store inside ./mlruns (the file store is deprecated in MLflow 3)."""
        self.mode = "local"
        root = Path(self.local_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.mlflow.set_tracking_uri(f"sqlite:///{root.as_posix()}/mlflow.db")
        name = self.cfg["experiment"]
        if self.mlflow.MlflowClient().get_experiment_by_name(name) is None:
            self.mlflow.create_experiment(name, artifact_location=(root / "artifacts").resolve().as_uri())
        self.mlflow.set_experiment(name)

    @contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None):
        """Start a run (parent, or nested child). Falls back to local if the server drops."""
        if not self.enabled:
            yield None
            return
        try:
            ctx = self.mlflow.start_run(run_name=name, nested=nested, tags=tags)
        except Exception as e:
            log.warning("could not start MLflow run on %s (%s). Switching to local.", self.mode, type(e).__name__)
            while self.mlflow.active_run():
                self.mlflow.end_run()
            self._use_local()
            ctx = self.mlflow.start_run(run_name=name, nested=False, tags=tags)
        with ctx as active:
            yield active

    def _safe(self, fn, *args, **kwargs) -> None:
        if not self.enabled:
            return
        try:
            fn(*args, **kwargs)
        except Exception as e:
            log.warning("MLflow call %s failed: %s: %s", getattr(fn, "__name__", "?"), type(e).__name__, str(e)[:120])

    def params(self, d: dict) -> None:
        if self.enabled:
            self._safe(self.mlflow.log_params, {k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d: dict, prefix: str = "") -> None:
        if self.enabled:
            self._safe(self.mlflow.log_metrics, flatten(d, prefix))

    def tags(self, d: dict) -> None:
        if self.enabled:
            self._safe(self.mlflow.set_tags, d)

    def artifact(self, path: str | Path, subdir: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._safe(self.mlflow.log_artifact, str(path), subdir)

    def dict_artifact(self, obj: dict, name: str) -> None:
        if self.enabled:
            self._safe(self.mlflow.log_dict, obj, name)

    def log_model_bundle(self, path: str | Path, registered_name: str) -> None:
        """Upload the model file to the active run and add it as a new version in the model registry."""
        if not self.enabled:
            return
        try:
            self.mlflow.log_artifact(str(path), "model_bundle")
            run = self.mlflow.active_run()
            client = self.mlflow.MlflowClient()
            try:
                client.create_registered_model(registered_name)
            except Exception:  # already exists
                pass
            version = client.create_model_version(
                registered_name, source=f"{run.info.artifact_uri}/model_bundle", run_id=run.info.run_id)
            log.info("registered %s version %s", registered_name, version.version)
        except Exception as e:
            log.warning("could not register model %s: %s: %s", registered_name, type(e).__name__, str(e)[:120])
