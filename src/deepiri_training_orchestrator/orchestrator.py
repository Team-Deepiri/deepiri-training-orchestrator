"""
Framework-agnostic training orchestrator: reproducibility + optional tracking + callbacks.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, TypeVar

from deepiri_training_orchestrator.callbacks import (
    CheckpointCallback,
    TorchCheckpointCallback,
    TrainingContext,
    compose_callbacks,
)
from deepiri_training_orchestrator.config import DatasetProvenance, TrainingRunConfig
from deepiri_training_orchestrator.datasets import prepare_training_run, version_dataset
from deepiri_training_orchestrator.distributed import DistributedContext, gather_metrics
from deepiri_training_orchestrator.reproducibility import ReproducibilityController
from deepiri_training_orchestrator.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT")


class EpochIterator:
    """Wrap a batch iterable and increment epoch on each exhaustion."""

    def __init__(self, batches: Iterable[BatchT]) -> None:
        self._batches = batches
        self.epoch = 0

    def __iter__(self) -> Iterator[BatchT]:
        self.epoch += 1
        return iter(self._batches)


class TrainingOrchestrator:
    """Runs a training loop with seeding, fingerprinting, tracking, and callbacks."""

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
        run_config: Optional[TrainingRunConfig] = None,
        dataset_provenance: Optional[DatasetProvenance] = None,
        correlation_id: Optional[str] = None,
        distributed_context: Optional[DistributedContext] = None,
    ) -> None:
        self.run_config = run_config
        if run_config is not None:
            config = {**dict(config), **run_config.hyperparameters}
            max_steps = run_config.max_steps
            log_every = run_config.log_every
            eval_every = run_config.eval_every or eval_every
            run_name = run_config.run_name or run_name
            correlation_id = run_config.correlation_id or correlation_id
            dataset_provenance = run_config.dataset or dataset_provenance
        self.config = dict(config)
        self.reproducibility = reproducibility
        self.max_steps = max_steps
        self.log_every = max(1, log_every)
        self.eval_every = eval_every
        self.experiment_tracker = experiment_tracker
        self.run_name = run_name
        self.callbacks = compose_callbacks(callbacks)
        self._code_hash = code_hash
        self._dataset_provenance = dataset_provenance
        self._correlation_id = correlation_id
        self._distributed = distributed_context
        self._ctx = TrainingContext(max_steps=max_steps)

        if run_config is not None and run_config.checkpoint.directory:
            extra = [
                CheckpointCallback(
                    run_config.checkpoint.directory,
                    every=run_config.checkpoint.every_n_steps,
                ),
            ]
            if run_config.checkpoint.save_state_dict:
                extra.append(
                    TorchCheckpointCallback(
                        run_config.checkpoint.directory,
                        every=run_config.checkpoint.every_n_steps,
                    )
                )
            self.callbacks = compose_callbacks((callbacks or []) + extra)

    def fit(
        self,
        batches: Iterable[BatchT],
        *,
        train_step: Callable[[int, BatchT], Dict[str, float]],
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        max_epochs: Optional[int] = None,
    ) -> TrainingContext:
        fingerprint = self.reproducibility.generate_training_fingerprint(
            self.config,
            code_hash=self._code_hash,
            dataset_hash=(
                self._dataset_provenance.content_hash if self._dataset_provenance else None
            ),
        )
        self._ctx.fingerprint = fingerprint
        self._ctx.max_steps = self.max_steps
        if self._correlation_id:
            self._ctx.extra["correlation_id"] = self._correlation_id

        tracker = self.experiment_tracker
        if tracker is not None:
            tracker.start_run(run_name=self.run_name)
            tracker.log_params({k: str(v) for k, v in self.config.items()})
            tracker.log_params({"training_fingerprint": fingerprint})
            if self.run_config and self.run_config.tracking.auto_log_git:
                tracker.log_git_info()
            if self.run_config and self.run_config.tracking.auto_log_code:
                tracker.log_code(self.run_config.tracking.code_path)
            if self._dataset_provenance is not None:
                tracker.log_dataset(
                    self._dataset_provenance.path,
                    dataset_hash=self._dataset_provenance.content_hash,
                )

        for cb in self.callbacks:
            cb.on_train_begin(self, self._ctx)

        step = 0
        epochs = 0
        batch_source: Iterable[BatchT] = batches
        try:
            while step < self.max_steps:
                if max_epochs is not None and epochs >= max_epochs:
                    break
                epoch_iter = EpochIterator(batch_source)
                batch_iter = iter(epoch_iter)
                epochs = epoch_iter.epoch
                self._ctx.epoch = epochs
                epoch_had_batch = False

                while step < self.max_steps:
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    epoch_had_batch = True

                    metrics = train_step(step, batch)
                    self._ctx.step = step

                    if tracker is not None and (
                        self.run_config is None or self.run_config.distributed.is_main_process
                    ):
                        tracker.log_metrics(metrics, step=step)

                    if step % self.log_every == 0:
                        logger.info("step=%s epoch=%s %s", step, epochs, metrics)

                    for cb in self.callbacks:
                        cb.on_step_end(self, self._ctx, metrics)

                    if self.eval_every and eval_fn and (step + 1) % self.eval_every == 0:
                        eval_metrics = eval_fn()
                        if self._distributed is not None:
                            eval_metrics = gather_metrics(self._distributed, eval_metrics)
                        self._ctx.extra["last_eval"] = eval_metrics
                        if tracker is not None:
                            em = {f"eval/{k}": v for k, v in eval_metrics.items()}
                            tracker.log_metrics(em, step=step)
                        for cb in self.callbacks:
                            cb.on_eval_end(self, self._ctx, eval_metrics)
                        if self._ctx.extra.get("stop_training"):
                            logger.info("Early stopping triggered at step %s", step)
                            return self._ctx

                    step += 1

                for cb in self.callbacks:
                    cb.on_train_epoch_end(self, self._ctx)

                if not epoch_had_batch:
                    break
                if max_epochs is None:
                    break
        except Exception as exc:
            for cb in self.callbacks:
                cb.on_exception(self, self._ctx, exc)
            raise
        finally:
            for cb in self.callbacks:
                cb.on_train_end(self, self._ctx)
            if tracker is not None:
                tracker.end_run()
            if self._dataset_provenance is not None and self.run_config:
                try:
                    version_dataset(
                        self._dataset_provenance.path,
                        dataset_name=self._dataset_provenance.dataset_id,
                    )
                except Exception as exc:
                    logger.warning("Could not version dataset after training: %s", exc)

        return self._ctx

    @classmethod
    def from_run_config(
        cls,
        run_config: TrainingRunConfig,
        *,
        experiment_tracker: Optional[ExperimentTracker] = None,
        callbacks: Optional[List[Any]] = None,
        code_hash: Optional[str] = None,
        distributed_context: Optional[DistributedContext] = None,
        auto_prepare_dataset: bool = True,
    ) -> TrainingOrchestrator:
        repro = ReproducibilityController(seed=run_config.seed)
        repro.set_seeds()
        dataset_provenance = run_config.dataset
        if auto_prepare_dataset and dataset_provenance and dataset_provenance.path:
            try:
                prepared = prepare_training_run(
                    dataset_provenance.path,
                    preset="training",
                    dataset_id=dataset_provenance.dataset_id,
                )
                dataset_provenance = prepared.provenance
            except Exception as exc:
                logger.warning("Auto dataset prep skipped: %s", exc)
        if experiment_tracker is None and run_config.tracking.mlflow_uri:
            experiment_tracker = ExperimentTracker(
                "training",
                tracking_uri=run_config.tracking.mlflow_uri,
                use_wandb=run_config.tracking.use_wandb,
                wandb_project=run_config.tracking.wandb_project,
            )
        return cls(
            run_config.hyperparameters,
            repro,
            run_config=run_config,
            experiment_tracker=experiment_tracker,
            callbacks=callbacks,
            code_hash=code_hash,
            dataset_provenance=dataset_provenance,
            correlation_id=run_config.correlation_id,
            distributed_context=distributed_context,
        )
