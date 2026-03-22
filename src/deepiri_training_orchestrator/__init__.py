"""
deepiri-training-orchestrator — reproducible training loops, experiment tracking, callbacks.
"""

from deepiri_training_orchestrator.callbacks import (
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TrainingContext,
    compose_callbacks,
)
from deepiri_training_orchestrator.orchestrator import TrainingOrchestrator
from deepiri_training_orchestrator.reproducibility import (
    ReproducibilityController,
    initialize_deterministic_training,
)
from deepiri_training_orchestrator.tracking import (
    DatasetVersioning,
    ExperimentTracker,
    ModelRegistry,
)

__all__ = [
    "CallbackList",
    "CheckpointCallback",
    "DatasetVersioning",
    "EarlyStoppingCallback",
    "ExperimentTracker",
    "LoggingCallback",
    "ModelRegistry",
    "ReproducibilityController",
    "TrainingContext",
    "TrainingOrchestrator",
    "compose_callbacks",
    "initialize_deterministic_training",
]

__version__ = "0.1.0"
