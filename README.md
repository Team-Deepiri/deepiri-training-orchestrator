# deepiri-training-orchestrator

Library for **training orchestration and experiment reproducibility**: global seeding (Python / NumPy / PyTorch / CUDA), deterministic CUDA options, **config fingerprinting**, optional **MLflow** and **Weights & Biases** logging, and a **callback-based training loop** that works with any framework—plug in your model and batch iterator, supply a `train_step(step, batch) -> metrics` function, and get consistent checkpoints and metadata.

This package extracts and generalizes patterns from **Helox** (`core/reproducibility_controller.py`, `mlops/infrastructure/experiment_tracker.py`) into a reusable installable module so new projects do not need a bespoke `train.py` for every repo.

## Install

```bash
pip install deepiri-training-orchestrator
```

Development:

```bash
git clone https://github.com/Team-Deepiri/deepiri-training-orchestrator.git
cd deepiri-training-orchestrator
poetry install
```

Dependencies include **PyTorch**, **NumPy**, and **MLflow**. **wandb** is optional (used only if installed and `use_wandb=True`).

## Quick start

```python
from deepiri_training_orchestrator import (
    ReproducibilityController,
    TrainingOrchestrator,
    ExperimentTracker,
)

repro = ReproducibilityController(seed=42)
repro.set_seeds()

tracker = ExperimentTracker(
    "my_experiment",
    tracking_uri="file:./mlruns",
)

orch = TrainingOrchestrator(
    {"lr": 1e-4, "epochs": 10},
    reproducibility=repro,
    max_steps=1000,
    log_every=50,
    experiment_tracker=tracker,
)

def train_step(step, batch):
    # Your framework here (PyTorch, etc.)
    return {"loss": 0.0}

orch.fit(my_batch_iterator(), train_step=train_step)
```

## Components

| Piece | Purpose |
|--------|---------|
| `ReproducibilityController` | Seeds, dataloader worker init, SHA256 config fingerprint, fingerprint verify |
| `TrainingOrchestrator` | Generic loop: `fit(batches, train_step=...)`, optional `eval_fn`, callbacks |
| `ExperimentTracker` | MLflow runs, params/metrics, dataset hash, artifact dirs (e.g. LoRA output) |
| `DatasetVersioning` | Optional DVC workflow helper |
| `ModelRegistry` | MLflow registry helpers |
| Callbacks | `LoggingCallback`, `CheckpointCallback`, `EarlyStoppingCallback` |

## Helox migration

Point **diri-helox** at this package for shared `ReproducibilityController` and `ExperimentTracker`, then keep Helox-specific trainers (e.g. `UnifiedTrainingOrchestrator`) as thin wrappers that compose this library with domain code.

## License

MIT — see [LICENSE](LICENSE).
