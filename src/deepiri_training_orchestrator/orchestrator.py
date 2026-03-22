"""
Framework-agnostic training orchestrator: reproducibility + optional tracking + callbacks.

Use with any stack (PyTorch, JAX, sklearn) by supplying a ``train_step`` callable
and an iterable of batches.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, TypeVar

from deepiri_training_orchestrator.callbacks import TrainingContext, compose_callbacks
from deepiri_training_orchestrator.reproducibility import ReproducibilityController
from deepiri_training_orchestrator.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT")


class TrainingOrchestrator:
    """
    Runs a training loop with consistent seeding, fingerprinting, optional MLflow/W&B,
    and callback hooks (checkpointing, early stopping, logging).

    Example::

        repro = ReproducibilityController(seed=42)
        repro.set_seeds()
        orch = TrainingOrchestrator(
            config={"lr": 1e-4},
            reproducibility=repro,
            max_steps=100,
            log_every=10,
        )
        orch.fit(batch_iter, train_step=lambda step, batch: {"loss": float(batch)})
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        reproducibility: ReproducibilityController,
        *,
        max_steps: int = 1000,
        log_every: int = 50,
        eval_every: Optional[int] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
        run_name: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        code_hash: Optional[str] = None,
    ) -> None:
        self.config = dict(config)
        self.reproducibility = reproducibility
        self.max_steps = max_steps
        self.log_every = max(1, log_every)
        self.eval_every = eval_every
        self.experiment_tracker = experiment_tracker
        self.run_name = run_name
        self.callbacks = compose_callbacks(callbacks)
        self._code_hash = code_hash
        self._ctx = TrainingContext(max_steps=max_steps)

    def fit(
        self,
        batches: Iterable[BatchT],
        *,
        train_step: Callable[[int, BatchT], Dict[str, float]],
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> TrainingContext:
        """
        Iterate over ``batches`` until ``max_steps`` is reached or the iterator is exhausted.

        ``train_step`` receives ``(global_step, batch)`` and returns a metrics dict
        (e.g. ``{"loss": 0.42}``).
        """
        fingerprint = self.reproducibility.generate_training_fingerprint(
            self.config,
            code_hash=self._code_hash,
        )
        self._ctx.fingerprint = fingerprint
        self._ctx.max_steps = self.max_steps

        tracker = self.experiment_tracker
        if tracker is not None:
            tracker.start_run(run_name=self.run_name)
            tracker.log_params({k: str(v) for k, v in self.config.items()})
            tracker.log_params({"training_fingerprint": fingerprint})

        for cb in self.callbacks:
            cb.on_train_begin(self, self._ctx)

        step = 0
        batch_iter: Iterator[BatchT] = iter(batches)
        try:
            while step < self.max_steps:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break

                metrics = train_step(step, batch)
                self._ctx.step = step

                if tracker is not None:
                    tracker.log_metrics(metrics, step=step)

                if step % self.log_every == 0:
                    logger.info("step=%s %s", step, metrics)

                for cb in self.callbacks:
                    cb.on_step_end(self, self._ctx, metrics)

                if self.eval_every and eval_fn and (step + 1) % self.eval_every == 0:
                    eval_metrics = eval_fn()
                    self._ctx.extra["last_eval"] = eval_metrics
                    if tracker is not None:
                        em = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        tracker.log_metrics(em, step=step)
                    for cb in self.callbacks:
                        cb.on_eval_end(self, self._ctx, eval_metrics)
                    if self._ctx.extra.get("stop_training"):
                        logger.info("Early stopping triggered at step %s", step)
                        break

                step += 1
        finally:
            for cb in self.callbacks:
                cb.on_train_end(self, self._ctx)
            if tracker is not None:
                tracker.end_run()

        return self._ctx
