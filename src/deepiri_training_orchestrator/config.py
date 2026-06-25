"""Pydantic configuration models for training runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    mlflow_uri: str = "file:./mlruns"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    auto_log_git: bool = True
    auto_log_code: bool = False
    code_path: str = "."


class CheckpointConfig(BaseModel):
    directory: Path = Path("./checkpoints")
    every_n_steps: int = 500
    save_state_dict: bool = False


class DistributedConfig(BaseModel):
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.local_rank == 0


class DatasetProvenance(BaseModel):
    """Dataset identity for a training run (from dataset-processor manifest)."""

    dataset_id: str
    version: str
    path: str
    content_hash: str
    row_count: int = 0
    produced_by: str = "deepiri-dataset-processor"


class TrainingRunConfig(BaseModel):
    seed: int = 1337
    max_steps: int = 1000
    log_every: int = 50
    eval_every: Optional[int] = None
    run_name: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    dataset: Optional[DatasetProvenance] = None
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    def flat_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "log_every": self.log_every,
        }
        if self.eval_every is not None:
            params["eval_every"] = self.eval_every
        if self.correlation_id:
            params["correlation_id"] = self.correlation_id
        params.update({f"hp.{k}": v for k, v in self.hyperparameters.items()})
        return params
