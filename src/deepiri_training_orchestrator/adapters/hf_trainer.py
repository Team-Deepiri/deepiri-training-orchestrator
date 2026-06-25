"""HuggingFace Trainer adapter for TrainingOrchestrator.fit()."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class HFTrainingAdapter:
    """
    Wraps a HuggingFace ``Trainer`` so orchestrator owns the training loop.

  The adapter drives ``training_step`` / ``prediction_step`` while respecting
  gradient accumulation configured on the trainer.
    """

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._micro_step = 0

    def train_step(self, step: int, batch: Any) -> Dict[str, float]:
        """Execute one micro-batch through the HF trainer."""
        self.trainer.model.train()
        batch = self.trainer._prepare_inputs(batch)
        with self.trainer.compute_loss_context_manager():
            loss = self.trainer.training_step(self.trainer.model, batch)
        self._micro_step += 1

        metrics: Dict[str, float] = {}
        if hasattr(loss, "item"):
            metrics["loss"] = float(loss.item())
        elif isinstance(loss, (int, float)):
            metrics["loss"] = float(loss)

        accum = getattr(self.trainer.args, "gradient_accumulation_steps", 1)
        if self._micro_step % accum == 0:
            if hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
                self.trainer.optimizer.step()
                self.trainer.optimizer.zero_grad(set_to_none=True)
            if hasattr(self.trainer, "lr_scheduler") and self.trainer.lr_scheduler is not None:
                self.trainer.lr_scheduler.step()

        metrics["learning_rate"] = float(
            self.trainer.optimizer.param_groups[0]["lr"]
            if getattr(self.trainer, "optimizer", None)
            else 0.0
        )
        return metrics

    def eval_fn(self) -> Dict[str, float]:
        """Run evaluation loop and return metrics."""
        if not hasattr(self.trainer, "evaluate"):
            return {}
        raw = self.trainer.evaluate()
        return {
            k.replace("eval_", ""): float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float))
        }

    def state_dict_fn(self) -> Callable[[], Dict[str, Any]]:
        """Return callable for checkpoint callbacks."""

        def _state() -> Dict[str, Any]:
            return {
                "model": self.trainer.model.state_dict(),
                "optimizer": (
                    self.trainer.optimizer.state_dict()
                    if getattr(self.trainer, "optimizer", None)
                    else None
                ),
            }

        return _state
