"""Distributed training context and helpers (DDP/FSDP via accelerate)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from deepiri_training_orchestrator.config import DistributedConfig

T = TypeVar("T")

try:
    from accelerate import Accelerator

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    Accelerator = None  # type: ignore[misc, assignment]


@dataclass
class DistributedContext:
    """Runtime distributed state for multi-GPU training."""

    config: DistributedConfig
    accelerator: Any = None

    @property
    def is_main_process(self) -> bool:
        if self.accelerator is not None:
            return bool(self.accelerator.is_main_process)
        return self.config.is_main_process

    @property
    def local_rank(self) -> int:
        if self.accelerator is not None:
            return int(self.accelerator.local_process_index)
        return self.config.local_rank

    @property
    def world_size(self) -> int:
        if self.accelerator is not None:
            return int(self.accelerator.num_processes)
        return self.config.world_size


def init_distributed(
    config: Optional[DistributedConfig] = None,
    *,
    mixed_precision: Optional[str] = None,
) -> DistributedContext:
    """Initialize accelerate Accelerator when available."""
    cfg = config or DistributedConfig(
        local_rank=int(os.getenv("LOCAL_RANK", "0")),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
    )
    accelerator = None
    if HAS_ACCELERATE:
        accelerator = Accelerator(mixed_precision=mixed_precision)
        cfg = DistributedConfig(
            local_rank=accelerator.local_process_index,
            world_size=accelerator.num_processes,
        )
    return DistributedContext(config=cfg, accelerator=accelerator)


def prepare_model_optimizer(
    ctx: DistributedContext,
    model: T,
    optimizer: Any,
) -> tuple[T, Any]:
    """Wrap model/optimizer with accelerate when distributed."""
    if ctx.accelerator is not None:
        return ctx.accelerator.prepare(model, optimizer)
    return model, optimizer


def main_process_only(ctx: DistributedContext, fn: Callable[[], T]) -> Optional[T]:
    """Run callable only on rank 0."""
    if ctx.is_main_process:
        return fn()
    return None


def gather_metrics(ctx: DistributedContext, metrics: Dict[str, float]) -> Dict[str, float]:
    """Placeholder for cross-rank metric reduction."""
    if ctx.accelerator is not None and hasattr(ctx.accelerator, "gather"):
        try:
            import torch

            tensors = {k: torch.tensor(v) for k, v in metrics.items()}
            gathered = ctx.accelerator.gather(tensors)
            if isinstance(gathered, dict):
                return {k: float(v.mean()) for k, v in gathered.items()}
        except Exception:
            pass
    return metrics
