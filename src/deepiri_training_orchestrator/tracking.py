"""
Experiment tracking with MLflow and optional Weights & Biases.
Extracted from Helox ``mlops/infrastructure/experiment_tracker.py`` (cleaned up).
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None  # type: ignore[misc, assignment]


def _setup_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    return lg


_tracker_log = _setup_logger("deepiri.training.tracker")


class ExperimentTracker:
    """Unified experiment tracking with MLflow and optional W&B."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.use_wandb = bool(use_wandb and wandb_project and HAS_WANDB)
        if use_wandb and wandb_project and not HAS_WANDB:
            _tracker_log.warning("wandb requested but not installed. Continuing without W&B.")
        elif self.use_wandb and wandb_project:
            wandb.init(project=wandb_project, name=experiment_name)

        self.client = MlflowClient()
        self.current_run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags or {})
        _tracker_log.info("Experiment run started: run_name=%s", run_name)
        return self.current_run

    def log_git_info(self) -> None:
        """Tag the active run with the current git commit hash when available."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            mlflow.set_tag("git_commit", commit)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            _tracker_log.debug("Could not log git info: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        safe = {k: str(v) for k, v in params.items()}
        mlflow.log_params(safe)
        if self.use_wandb and HAS_WANDB:
            wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)
        if self.use_wandb and HAS_WANDB:
            wandb.log(metrics, step=step)

    def log_dataset(self, dataset_path: str, dataset_hash: Optional[str] = None) -> None:
        if dataset_hash is None:
            dataset_hash = self._compute_dataset_hash(dataset_path)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_hash", dataset_hash)
        if self.use_wandb and HAS_WANDB:
            wandb.config.update({"dataset_path": dataset_path, "dataset_hash": dataset_hash})

    def log_model(self, model_or_dir: Any, artifact_path: str = "model") -> None:
        """
        Log a saved model directory as MLflow artifacts (recommended for LoRA/PEFT).
        Pass the folder where ``save_pretrained()`` wrote files.
        """
        if isinstance(model_or_dir, (str, os.PathLike)) and os.path.isdir(model_or_dir):
            mlflow.log_artifacts(str(model_or_dir), artifact_path=artifact_path)
            return
        raise ValueError(
            "ExperimentTracker.log_model expected a directory path. "
            "Pass the folder where save_pretrained() wrote adapter/tokenizer files."
        )

    def log_code(self, code_path: str = ".") -> None:
        try:
            mlflow.log_artifacts(code_path, "code")
        except Exception as e:
            _tracker_log.warning("Code logging failed: %s", e)

    def end_run(self, status: str = "FINISHED") -> None:
        if self.current_run:
            mlflow.end_run(status=status)
            if self.use_wandb and HAS_WANDB:
                wandb.finish()
            _tracker_log.info("Experiment run ended: status=%s", status)

    def _compute_dataset_hash(self, dataset_path: str) -> str:
        path = Path(dataset_path)
        sha256 = hashlib.sha256()
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    rel = str(p.relative_to(path)).encode()
                    sha256.update(rel)
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256.update(chunk)
            return sha256.hexdigest()
        raise FileNotFoundError(dataset_path)

    def register_model(self, run_id: str, model_name: str, stage: str = "Staging") -> None:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        version = int(result.version)
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        _tracker_log.info("Model registered: %s version=%s stage=%s", model_name, version, stage)


class DatasetVersioning:
    """Optional DVC-based dataset versioning (requires ``dvc`` and a DVC repo)."""

    def __init__(self, dvc_repo_path: str = ".") -> None:
        self.dvc_repo_path = Path(dvc_repo_path)

    def version_dataset(self, dataset_path: str, description: str = "") -> None:
        del description
        import subprocess

        dvc_path = self.dvc_repo_path / "data" / Path(dataset_path).name
        subprocess.run(
            ["dvc", "add", str(dvc_path), "-f", str(dvc_path.with_suffix(".dvc"))],
            check=True,
        )
        dvc_file = str(dvc_path.with_suffix(".dvc"))
        subprocess.run(["git", "add", dvc_file, dvc_file + ".gitignore"], check=True)
        _tracker_log.info("Dataset versioned: %s", dataset_path)


class ModelRegistry:
    """Thin helpers around MLflow model registry."""

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        self.client = MlflowClient(tracking_uri)

    def list_models(self, filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
        models = self.client.search_registered_models(filter_string=filter_string)
        return [{"name": m.name, "versions": len(m.latest_versions)} for m in models]

    def get_latest_model(self, model_name: str, stage: str = "Production") -> Optional[str]:
        try:
            model = self.client.get_latest_versions(model_name, stages=[stage])
            if model:
                return model[0].source
        except Exception as e:
            logger.error("Model retrieval failed: %s", e)
        return None

    def promote_model(self, model_name: str, version: int, stage: str) -> None:
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        _tracker_log.info("Model promoted: %s v%s -> %s", model_name, version, stage)
