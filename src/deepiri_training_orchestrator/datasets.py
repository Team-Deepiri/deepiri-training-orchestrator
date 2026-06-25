"""Dataset preparation facade — delegates to deepiri-dataset-processor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner, clean_text_document
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
from deepiri_dataset_processor.deduplication.semantic_dedup import SemanticDeduplicationEngine
from deepiri_dataset_processor.manifest import build_manifest
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline
from deepiri_dataset_processor.pipeline.stages import DataCleaningStage, DataValidationStage
from deepiri_dataset_processor.safety.leakage_detector import DataLeakageDetector
from deepiri_dataset_processor.versioning.filesystem import DatasetVersioningSystem

from deepiri_training_orchestrator.config import DatasetProvenance

ManifestLike = Union[Dict[str, Any], Any]


def prepare_dataset(
    dataset_path: Path | str,
    *,
    clean: bool = True,
    validate: bool = True,
) -> Any:
    """Run the dataset-processor pipeline stages on in-memory or file-backed data."""
    path = Path(dataset_path)
    stages = []
    if clean:
        stages.append(DataCleaningStage())
    if validate:
        stages.append(DataValidationStage(config={"required_fields": ["text"]}))
    if not stages:
        return path
    pipeline = DatasetPipeline(stages=stages)
    if path.is_file() and path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            import json

            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pipeline.run(records)
    return pipeline.run(path)


def version_dataset(dataset_path: Path | str, *, dataset_name: Optional[str] = None) -> str:
    """Create a versioned snapshot; returns version id."""
    versioning = DatasetVersioningSystem()
    name = dataset_name or Path(dataset_path).stem
    meta = versioning.create_dataset_version(Path(dataset_path), name)
    return str(meta.get("version", ""))


def detect_leakage(
    train_texts: List[str],
    eval_texts: List[str],
    *,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """Detect train/eval leakage via dataset-processor."""
    detector = DataLeakageDetector(overlap_threshold=threshold)
    return detector.detect_train_eval_contamination(train_texts, eval_texts)


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Exact deduplication on text strings."""
    return ExactDeduplicator().filter_duplicates(texts)


def semantic_deduplicate(texts: List[str], *, threshold: float = 0.95) -> List[str]:
    """Semantic deduplication on text strings."""
    engine = SemanticDeduplicationEngine(similarity_threshold=threshold)
    return engine.filter_duplicates(texts)


def build_dataset_manifest(
    dataset_path: Path | str,
    *,
    dataset_id: Optional[str] = None,
    version: Optional[str] = None,
) -> ManifestLike:
    """Build a DatasetManifest via dataset-processor."""
    return build_manifest(dataset_path, dataset_id=dataset_id, version=version)


def provenance_from_manifest(manifest: ManifestLike) -> DatasetProvenance:
    """Convert a manifest object or dict into DatasetProvenance."""
    if hasattr(manifest, "model_dump"):
        data = manifest.model_dump(by_alias=True)
        return DatasetProvenance(
            dataset_id=data["id"],
            version=data["version"],
            path=data["path"],
            content_hash=data["content_hash"],
            row_count=data.get("row_count", 0),
            produced_by=data.get("produced_by", "deepiri-dataset-processor"),
        )
    if hasattr(manifest, "id"):
        return DatasetProvenance(
            dataset_id=manifest.id,
            version=manifest.version,
            path=manifest.path,
            content_hash=manifest.content_hash,
            row_count=getattr(manifest, "row_count", 0),
            produced_by=getattr(manifest, "produced_by", "deepiri-dataset-processor"),
        )
    return DatasetProvenance(
        dataset_id=manifest["id"],
        version=manifest["version"],
        path=manifest["path"],
        content_hash=manifest["content_hash"],
        row_count=manifest.get("row_count", 0),
        produced_by=manifest.get("produced_by", "deepiri-dataset-processor"),
    )


def clean_text(text: str) -> Optional[str]:
    """Clean a single text document."""
    return clean_text_document(text)


__all__ = [
    "TextCleaner",
    "build_dataset_manifest",
    "clean_text",
    "deduplicate_texts",
    "detect_leakage",
    "prepare_dataset",
    "provenance_from_manifest",
    "semantic_deduplicate",
    "version_dataset",
]
