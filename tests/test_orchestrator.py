import tempfile
from pathlib import Path

from deepiri_training_orchestrator import (
    CheckpointCallback,
    ExperimentTracker,
    LoggingCallback,
    ReproducibilityController,
    TrainingOrchestrator,
)


def test_fit_loop() -> None:
    repro = ReproducibilityController(seed=0)
    repro.set_seeds()
    orch = TrainingOrchestrator(
        {"lr": 0.01},
        reproducibility=repro,
        max_steps=5,
        log_every=2,
    )

    def batches():
        for i in range(10):
            yield float(i)

    def train_step(step, batch):
        return {"loss": batch * 0.1}

    ctx = orch.fit(batches(), train_step=train_step)
    assert ctx.step == 4


def test_fit_with_mlflow_file_store() -> None:
    tmp = tempfile.mkdtemp()
    uri = Path(tmp).as_uri()
    tracker = ExperimentTracker("orch_test", tracking_uri=uri)
    repro = ReproducibilityController(seed=1)
    repro.set_seeds()
    orch = TrainingOrchestrator(
        {"lr": 1e-3},
        reproducibility=repro,
        max_steps=3,
        experiment_tracker=tracker,
        callbacks=[LoggingCallback(every=1)],
    )

    ctx = orch.fit(range(100), train_step=lambda s, b: {"loss": float(s)})
    assert ctx.fingerprint is not None


def test_checkpoint_callback(tmp_path: Path) -> None:
    repro = ReproducibilityController(seed=2)
    repro.set_seeds()
    ck = CheckpointCallback(directory=tmp_path, every=2)
    orch = TrainingOrchestrator(
        {},
        reproducibility=repro,
        max_steps=5,
        callbacks=[ck],
    )
    orch.fit(range(20), train_step=lambda s, b: {"loss": 1.0})
    files = list(tmp_path.glob("checkpoint_step_*.json"))
    assert len(files) >= 1
