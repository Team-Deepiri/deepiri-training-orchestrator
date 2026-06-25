"""
deepiri-training-orchestrator — reproducible training loops, experiment tracking, callbacks.
"""

from deepiri_training_orchestrator.callbacks import (
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TorchCheckpointCallback,
    TrainingContext,
    compose_callbacks,
)
from deepiri_training_orchestrator.config import (
    CheckpointConfig,
    DatasetProvenance,
    DistributedConfig,
    TrackingConfig,
    TrainingRunConfig,
)
from deepiri_training_orchestrator.datasets import (
    build_dataset_manifest,
    clean_text,
    deduplicate_texts,
    detect_leakage,
    prepare_dataset,
    provenance_from_manifest,
    semantic_deduplicate,
    version_dataset,
)
from deepiri_training_orchestrator.feedback import (
    FeedbackBuffer,
    FeedbackLoopTrainer,
    corrections_to_manifest,
)
from deepiri_training_orchestrator.orchestrator import EpochIterator, TrainingOrchestrator
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
    "CheckpointConfig",
    "DatasetProvenance",
    "DatasetVersioning",
    "DistributedConfig",
    "EarlyStoppingCallback",
    "EpochIterator",
    "ExperimentTracker",
    "FeedbackBuffer",
    "FeedbackLoopTrainer",
    "LoggingCallback",
    "ModelRegistry",
    "ReproducibilityController",
    "TorchCheckpointCallback",
    "TrackingConfig",
    "TrainingContext",
    "TrainingOrchestrator",
    "TrainingRunConfig",
    "build_dataset_manifest",
    "clean_text",
    "compose_callbacks",
    "corrections_to_manifest",
    "deduplicate_texts",
    "detect_leakage",
    "initialize_deterministic_training",
    "prepare_dataset",
    "provenance_from_manifest",
    "semantic_deduplicate",
    "version_dataset",
]

__version__ = "0.2.0"
