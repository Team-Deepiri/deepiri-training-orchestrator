"""Dataset preparation facade — delegates to deepiri-dataset-processor presets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from deepiri_dataset_processor import (
    build_manifest,
    feedback_preset,
    production_preset,
    training_preset,
)
from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner, clean_text_document
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
from deepiri_dataset_processor.deduplication.semantic_dedup import SemanticDeduplicationEngine
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline
from deepiri_dataset_processor.quality.checker import QualityChecker, QualityReport
from deepiri_dataset_processor.safety.leakage_detector import DataLeakageDetector
from deepiri_dataset_processor.streaming.chunked_jsonl import (
    load_jsonl_records,
    write_jsonl_records,
)
from deepiri_dataset_processor.versioning.filesystem import DatasetVersioningSystem

from deepiri_training_orchestrator.config import DatasetProvenance

ManifestLike = Union[Dict[str, Any], Any]
PresetName = Literal["training", "feedback", "production"]


@dataclass
class PreparedDataset:
    """Result of a full dataset preparation run."""

    path: Path
    manifest: ManifestLike
    provenance: DatasetProvenance
    records: List[Dict[str, Any]] = field(default_factory=list)
    quality_report: Optional[Dict[str, Any]] = None
    leakage_report: Optional[Dict[str, Any]] = None
    version_id: Optional[str] = None


def _pipeline_for_preset(
    preset: PresetName,
    *,
    eval_texts: Optional[List[str]] = None,
    dataset_id: Optional[str] = None,
) -> DatasetPipeline:
    if preset == "feedback":
        return feedback_preset(dataset_id=dataset_id)
    if preset == "production":
        return production_preset(eval_texts=eval_texts, dataset_id=dataset_id)
    return training_preset(required_fields=["text"], dataset_id=dataset_id)


def prepare_training_run(
    dataset_path: Path | str,
    *,
    preset: PresetName = "training",
    output_dir: Optional[Path | str] = None,
    eval_path: Optional[Path | str] = None,
    dataset_id: Optional[str] = None,
    run_leakage_check: bool = True,
    run_quality_gate: bool = False,
    quality_threshold: float = 0.8,
) -> PreparedDataset:
    """
    High-level dataset prep using dataset-processor presets.

    Runs the selected preset pipeline, optionally checks leakage against an
    eval split, and returns manifest + provenance for orchestrator runs.
    """
    path = Path(dataset_path)
    eval_texts: Optional[List[str]] = None
    if eval_path:
        eval_records = load_jsonl_records(eval_path) if Path(eval_path).exists() else []
        eval_texts = [str(r.get("text", "")) for r in eval_records]

    if path.is_file() or path.is_dir():
        pipeline = _pipeline_for_preset(preset, eval_texts=eval_texts, dataset_id=dataset_id)
        if path.suffix == ".jsonl" or path.is_dir():
            records = load_jsonl_records(path)
            result = pipeline.run(records)
        else:
            result = pipeline.run(path)
        if not result.success:
            raise RuntimeError(f"Dataset pipeline failed: {result.error}")
        processed = result.processed_data
        records = processed.data if hasattr(processed, "data") else processed
        metadata = getattr(processed, "metadata", {}) or {}
    else:
        raise FileNotFoundError(path)

    out_dir = Path(output_dir) if output_dir else path.parent / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_id or path.stem}.jsonl"
    write_jsonl_records(out_path, records)

    manifest = metadata.get("manifest") or build_manifest(
        out_path, dataset_id=dataset_id or path.stem
    )
    provenance = provenance_from_manifest(manifest)

    leakage_report = None
    if run_leakage_check and eval_texts:
        train_texts = [str(r.get("text", "")) for r in records]
        leakage_report = detect_leakage(train_texts, eval_texts)

    quality_report_dict = metadata.get("quality_report")
    if run_quality_gate and not quality_report_dict:
        report = run_quality_gate_check(manifest, out_path, threshold=quality_threshold)
        quality_report_dict = report.to_dict()

    version_id = None
    if metadata.get("version"):
        version_id = str(metadata["version"].get("version", ""))

    return PreparedDataset(
        path=out_path,
        manifest=manifest,
        provenance=provenance,
        records=records if isinstance(records, list) else [],
        quality_report=quality_report_dict,
        leakage_report=leakage_report,
        version_id=version_id,
    )


def run_quality_gate_check(
    manifest: ManifestLike,
    dataset_path: Path | str,
    *,
    threshold: float = 0.8,
) -> QualityReport:
    """Run quality checker and raise if below threshold."""
    records = load_jsonl_records(dataset_path)
    dataset_id = manifest.id if hasattr(manifest, "id") else manifest.get("id", "dataset")
    checker = QualityChecker()
    report = checker.check_quality(records, dataset_id=str(dataset_id))
    if report.overall_score < threshold:
        raise ValueError(
            f"Quality gate failed: score {report.overall_score:.2f} < {threshold}"
        )
    return report


def prepare_dataset(
    dataset_path: Path | str,
    *,
    preset: PresetName = "training",
    clean: bool = True,
    validate: bool = True,
) -> Any:
    """Backward-compatible prepare; delegates to preset pipeline."""
    del clean, validate
    prepared = prepare_training_run(dataset_path, preset=preset)
    return prepared.records or prepared.path


def version_dataset(dataset_path: Path | str, *, dataset_name: Optional[str] = None) -> str:
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
    detector = DataLeakageDetector(overlap_threshold=threshold)
    return detector.detect_train_eval_contamination(train_texts, eval_texts)


def deduplicate_texts(texts: List[str]) -> List[str]:
    return ExactDeduplicator().filter_duplicates(texts)


def semantic_deduplicate(texts: List[str], *, threshold: float = 0.95) -> List[str]:
    engine = SemanticDeduplicationEngine(similarity_threshold=threshold)
    return engine.filter_duplicates(texts)


def build_dataset_manifest(
    dataset_path: Path | str,
    *,
    dataset_id: Optional[str] = None,
    version: Optional[str] = None,
) -> ManifestLike:
    return build_manifest(dataset_path, dataset_id=dataset_id, version=version)


def provenance_from_manifest(manifest: ManifestLike) -> DatasetProvenance:
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


def build_run_provenance(manifest: ManifestLike) -> DatasetProvenance:
    """Alias for provenance_from_manifest."""
    return provenance_from_manifest(manifest)


def clean_text(text: str) -> Optional[str]:
    return clean_text_document(text)


__all__ = [
    "PreparedDataset",
    "TextCleaner",
    "build_dataset_manifest",
    "build_run_provenance",
    "clean_text",
    "deduplicate_texts",
    "detect_leakage",
    "prepare_dataset",
    "prepare_training_run",
    "provenance_from_manifest",
    "run_quality_gate_check",
    "semantic_deduplicate",
    "version_dataset",
]
