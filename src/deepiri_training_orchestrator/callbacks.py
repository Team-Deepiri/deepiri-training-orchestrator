"""
Callback hooks for training loops (checkpointing, early stopping, logging).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class TrainingContext:
    """Mutable state passed through callback hooks."""

    step: int = 0
    epoch: int = 0
    max_steps: int = 0
    fingerprint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingCallback(Protocol):
    def on_train_begin(self, orchestrator: Any, ctx: TrainingContext) -> None: ...

    def on_step_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None: ...

    def on_eval_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None: ...

    def on_train_end(self, orchestrator: Any, ctx: TrainingContext) -> None: ...


class CallbackList:
    """Default implementations for optional hook methods."""

    def on_train_begin(self, orchestrator: Any, ctx: TrainingContext) -> None:
        pass

    def on_step_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_eval_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None:
        pass

    def on_train_end(self, orchestrator: Any, ctx: TrainingContext) -> None:
        pass


class LoggingCallback(CallbackList):
    """Log metrics to the standard logger on an interval."""

    def __init__(self, every: int = 50) -> None:
        self.every = max(1, every)

    def on_step_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None:
        if ctx.step % self.every == 0:
            logger.info("step=%s metrics=%s", ctx.step, metrics)


class CheckpointCallback(CallbackList):
    """Write lightweight JSON checkpoints (metrics + fingerprint)."""

    def __init__(self, directory: Path, every: int = 500) -> None:
        self.directory = Path(directory)
        self.every = max(1, every)

    def on_step_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None:
        if ctx.step == 0 or ctx.step % self.every != 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": ctx.step,
            "metrics": metrics,
            "fingerprint": ctx.fingerprint,
        }
        path = self.directory / f"checkpoint_step_{ctx.step}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        orchestrator.reproducibility.save_fingerprint(self.directory / f"fp_step_{ctx.step}.json")


class EarlyStoppingCallback(CallbackList):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._bad_epochs = 0

    def on_eval_end(
        self,
        orchestrator: Any,
        ctx: TrainingContext,
        metrics: Dict[str, float],
    ) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return
        if self._best is None:
            self._best = value
            return
        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best = value
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        if self._bad_epochs >= self.patience:
            ctx.extra["stop_training"] = True


def compose_callbacks(callbacks: Optional[List[Any]]) -> List[Any]:
    return list(callbacks) if callbacks else []
