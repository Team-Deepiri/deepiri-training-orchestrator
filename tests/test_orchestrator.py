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


def test_best_checkpoint_callback_writes_marker_and_extra(tmp_path: Path) -> None:
    import json

    from deepiri_training_orchestrator import (
        BestCheckpointCallback,
        ReproducibilityController,
        TrainingOrchestrator,
    )

    state = {"step": 0}
    losses = {1: 1.0, 3: 0.5, 5: 0.7}

    def train_step(step: int, batch) -> dict:
        state["step"] = step
        return {"loss": 0.0}

    repro = ReproducibilityController(seed=4)
    repro.set_seeds()
    best = BestCheckpointCallback(directory=tmp_path, monitor="loss", mode="min")
    orch = TrainingOrchestrator(
        {},
        reproducibility=repro,
        max_steps=6,
        eval_every=2,
        callbacks=[best],
    )
    ctx = orch.fit(
        range(100),
        train_step=train_step,
        eval_fn=lambda: {"loss": losses[state["step"]]},
    )
    assert best._best == 0.5
    assert ctx.extra["best_step"] == 3
    assert ctx.extra["best_value"] == 0.5
    payload = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert payload["step"] == 3
    assert payload["value"] == 0.5


def test_best_checkpoint_state_dict_fn(tmp_path: Path) -> None:
    from deepiri_training_orchestrator import (
        BestCheckpointCallback,
        ReproducibilityController,
        TrainingOrchestrator,
    )

    calls = {"n": 0}

    def fake_state():
        calls["n"] += 1
        return {"w": [calls["n"]]}

    repro = ReproducibilityController(seed=5)
    repro.set_seeds()
    best = BestCheckpointCallback(
        directory=tmp_path,
        monitor="acc",
        mode="max",
        state_dict_fn=fake_state,
    )
    orch = TrainingOrchestrator(
        {},
        reproducibility=repro,
        max_steps=4,
        eval_every=2,
        callbacks=[best],
    )
    accs = {1: 0.6, 3: 0.9}
    state2 = {"step": 0}

    def train_step2(step: int, batch) -> dict:
        state2["step"] = step
        return {"acc": 0.0}

    orch.fit(
        range(100),
        train_step=train_step2,
        eval_fn=lambda: {"acc": accs[state2["step"]]},
    )
    assert (tmp_path / "best_state.pt").exists()
    assert calls["n"] == 2


def test_best_checkpoint_invalid_mode() -> None:
    import pytest

    from deepiri_training_orchestrator import BestCheckpointCallback

    with pytest.raises(ValueError):
        BestCheckpointCallback(directory=Path("/tmp"), mode="down")
