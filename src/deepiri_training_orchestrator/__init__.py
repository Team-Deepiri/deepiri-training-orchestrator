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
from deepiri_training_orchestrator.distributed import (
    DistributedContext,
    gather_metrics,
    init_distributed,
    main_process_only,
    prepare_model_optimizer,
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
    "DistributedContext",
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
    "gather_metrics",
    "init_distributed",
    "initialize_deterministic_training",
    "main_process_only",
    "prepare_dataset",
    "prepare_model_optimizer",
    "provenance_from_manifest",
    "semantic_deduplicate",
    "version_dataset",
]

__version__ = "0.3.0"
