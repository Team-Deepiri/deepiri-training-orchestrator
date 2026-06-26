"""Agent feedback-loop training: accumulate corrections, trigger mini fine-tunes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from deepiri_training_orchestrator.callbacks import TrainingContext
from deepiri_training_orchestrator.datasets import (
    clean_text,
    deduplicate_texts,
    prepare_training_run,
)
from deepiri_training_orchestrator.orchestrator import TrainingOrchestrator
from deepiri_training_orchestrator.reproducibility import ReproducibilityController


@dataclass
class LiveFineTuneConfig:
    """Configuration for live correction fine-tuning."""

    min_examples: int = 8
    max_steps: int = 100
    priority: str = "live"
    output_dir: str = "./feedback_corrections"
    seed: int = 1337


class FeedbackBuffer:
    """Buffer of training examples from agent corrections with JSONL persistence."""

    def __init__(self, *, min_examples: int = 8, persist_path: Optional[str] = None) -> None:
        self.min_examples = min_examples
        self.persist_path = Path(persist_path) if persist_path else None
        self._examples: List[Dict[str, Any]] = []
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self._examples.append(json.loads(line))

    def add(self, example: Mapping[str, Any]) -> None:
        record = dict(example)
        if "text" in record and isinstance(record["text"], str):
            cleaned = clean_text(record["text"])
            if cleaned:
                record["text"] = cleaned
        self._examples.append(record)
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def ready(self, min_examples: Optional[int] = None) -> bool:
        threshold = min_examples if min_examples is not None else self.min_examples
        return len(self._examples) >= threshold

    def flush(self) -> List[Dict[str, Any]]:
        texts = [str(ex.get("text", "")) for ex in self._examples]
        unique_texts = deduplicate_texts(texts)
        text_set = set(unique_texts)
        deduped = [ex for ex in self._examples if str(ex.get("text", "")) in text_set]
        self._examples.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
        return deduped

    def as_batches(self, batch_size: int = 4) -> Iterable[List[Dict[str, Any]]]:
        records = self.flush()
        for i in range(0, len(records), batch_size):
            yield records[i : i + batch_size]

    def __len__(self) -> int:
        return len(self._examples)


class FeedbackLoopTrainer:
    """Accumulates corrections; runs a mini training loop when the buffer is ready."""

    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        buffer: Optional[FeedbackBuffer] = None,
        *,
        min_examples: int = 8,
        live_config: Optional[LiveFineTuneConfig] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.live_config = live_config or LiveFineTuneConfig(min_examples=min_examples)
        self.buffer = buffer or FeedbackBuffer(min_examples=self.live_config.min_examples)

    def submit(
        self,
        artifact: Mapping[str, Any],
        *,
        train_step: Callable[[int, Any], Dict[str, float]],
        min_examples: Optional[int] = None,
    ) -> Optional[TrainingContext]:
        """Accept a LearningArtifact-shaped dict (text, metadata, etc.)."""
        normalized = dict(artifact)
        if "corrected_value" in normalized and "text" not in normalized:
            normalized["text"] = str(normalized["corrected_value"])
        self.buffer.add(normalized)
        if not self.buffer.ready(min_examples):
            return None
        batches = self.buffer.as_batches()
        return self.orchestrator.fit(batches, train_step=train_step)

    @classmethod
    def create_default(
        cls,
        config: Mapping[str, Any],
        *,
        live_config: Optional[LiveFineTuneConfig] = None,
    ) -> FeedbackLoopTrainer:
        cfg = live_config or LiveFineTuneConfig()
        repro = ReproducibilityController(seed=cfg.seed)
        repro.set_seeds()
        orch = TrainingOrchestrator(
            dict(config), reproducibility=repro, max_steps=cfg.max_steps
        )
        return cls(orch, live_config=cfg)

    def build_manifest_from_buffer(self, output_dir: Optional[str] = None) -> Any:
        """Flush buffer and build manifest via feedback preset pipeline."""
        examples = self.buffer.flush()
        return corrections_to_manifest(examples, output_dir or self.live_config.output_dir)


def corrections_to_manifest(
    examples: List[Mapping[str, Any]],
    output_dir: str,
) -> Any:
    """Write corrections to JSONL, run feedback preset, return provenance."""
    path = Path(output_dir) / "corrections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(dict(ex)) + "\n")
    prepared = prepare_training_run(path, preset="feedback", dataset_id="agent-corrections")
    return prepared.provenance
