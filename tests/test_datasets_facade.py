"""Tests for prepare_training_run facade."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    records = [
        {"text": "This is a long enough training document for the cleaner to accept it."},
        {"text": "Another unique document that passes minimum length requirements easily."},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_prepare_training_run(sample_jsonl: Path, tmp_path: Path):
    from deepiri_training_orchestrator.datasets import prepare_training_run

    prepared = prepare_training_run(
        sample_jsonl,
        preset="training",
        output_dir=tmp_path / "out",
        dataset_id="testds",
    )
    assert prepared.path.exists()
    assert prepared.provenance.dataset_id == "testds"
    assert prepared.provenance.content_hash


def test_fingerprint_includes_dataset_hash():
    from deepiri_training_orchestrator.config import DatasetProvenance
    from deepiri_training_orchestrator.reproducibility import ReproducibilityController

    repro = ReproducibilityController(seed=1)
    fp1 = repro.generate_training_fingerprint({"lr": 1e-4}, dataset_hash="abc")
    fp2 = repro.generate_training_fingerprint({"lr": 1e-4}, dataset_hash="def")
    assert fp1 != fp2


def test_corrections_to_manifest(tmp_path: Path):
    from deepiri_training_orchestrator.feedback import corrections_to_manifest

    examples = [{"text": "user correction one"}, {"text": "user correction two"}]
    prov = corrections_to_manifest(examples, str(tmp_path))
    assert prov.dataset_id == "agent-corrections"


def test_hf_adapter_train_step():
    from deepiri_training_orchestrator.adapters import HFTrainingAdapter

    class FakeOptimizer:
        param_groups = [{"lr": 0.001}]

        def step(self):
            pass

        def zero_grad(self, set_to_none=False):
            pass

    class FakeModel:
        def train(self):
            return self

    class FakeTrainer:
        args = type("A", (), {"gradient_accumulation_steps": 1})()
        model = FakeModel()
        optimizer = FakeOptimizer()
        lr_scheduler = None

        def _prepare_inputs(self, batch):
            return batch

        def compute_loss_context_manager(self):
            from contextlib import nullcontext

            return nullcontext()

        def training_step(self, model, batch):
            class L:
                def item(self):
                    return 0.5

            return L()

    adapter = HFTrainingAdapter(FakeTrainer())
    metrics = adapter.train_step(0, {"input_ids": [1, 2]})
    assert "loss" in metrics
