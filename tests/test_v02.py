from pathlib import Path

import pytest

from deepiri_training_orchestrator import (
    DatasetProvenance,
    FeedbackBuffer,
    FeedbackLoopTrainer,
    ReproducibilityController,
    TrainingOrchestrator,
    TrainingRunConfig,
    build_dataset_manifest,
    clean_text,
    deduplicate_texts,
    provenance_from_manifest,
)


def test_training_run_config_flat_params():
    cfg = TrainingRunConfig(seed=1, hyperparameters={"lr": 1e-4})
    params = cfg.flat_params()
    assert params["seed"] == 1
    assert params["hp.lr"] == 1e-4


def test_clean_text():
    long_text = "  hello   world  " + ("x" * 60)
    result = clean_text(long_text)
    assert result is not None
    assert "hello world" in result


def test_deduplicate_texts():
    assert deduplicate_texts(["a", "a", "b"]) == ["a", "b"]


def test_build_manifest_jsonl(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8")
    manifest = build_dataset_manifest(p, dataset_id="test")
    prov = provenance_from_manifest(manifest)
    assert isinstance(prov, DatasetProvenance)
    assert prov.dataset_id == "test"
    assert prov.row_count == 2


def test_feedback_buffer_and_trainer():
    repro = ReproducibilityController(seed=0)
    repro.set_seeds()
    orch = TrainingOrchestrator({"lr": 1e-4}, reproducibility=repro, max_steps=10)
    trainer = FeedbackLoopTrainer(orch, min_examples=2)
    assert trainer.submit({"text": "one"}, train_step=lambda s, b: {"loss": 1.0}) is None
    ctx = trainer.submit({"text": "two"}, train_step=lambda s, b: {"loss": 0.5})
    assert ctx is not None
    assert ctx.step >= 0


def test_orchestrator_from_run_config(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "x"}\n', encoding="utf-8")
    manifest = build_dataset_manifest(p)
    prov = provenance_from_manifest(manifest)
    cfg = TrainingRunConfig(
        max_steps=2,
        log_every=1,
        dataset=prov,
        tracking={"mlflow_uri": f"file:{tmp_path}/mlruns", "auto_log_git": False},
    )
    orch = TrainingOrchestrator.from_run_config(cfg)
    ctx = orch.fit([{"x": 1}, {"x": 2}], train_step=lambda s, b: {"loss": float(s)})
    assert ctx.fingerprint is not None
