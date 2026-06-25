"""Agent feedback-loop training: accumulate corrections, trigger mini fine-tunes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator

from deepiri_training_orchestrator.callbacks import TrainingContext
from deepiri_training_orchestrator.datasets import build_dataset_manifest, provenance_from_manifest
from deepiri_training_orchestrator.orchestrator import TrainingOrchestrator
from deepiri_training_orchestrator.reproducibility import ReproducibilityController


class FeedbackBuffer:
    """In-memory buffer of training examples from agent corrections."""

    def __init__(self, *, min_examples: int = 8) -> None:
        self.min_examples = min_examples
        self._examples: List[Dict[str, Any]] = []
        self._cleaner = TextCleaner()

    def add(self, example: Mapping[str, Any]) -> None:
        record = dict(example)
        if "text" in record and isinstance(record["text"], str):
            record["text"] = self._cleaner.clean(record["text"]) or record["text"]
        self._examples.append(record)

    def ready(self, min_examples: Optional[int] = None) -> bool:
        threshold = min_examples if min_examples is not None else self.min_examples
        return len(self._examples) >= threshold

    def flush(self) -> List[Dict[str, Any]]:
        texts = [str(ex.get("text", "")) for ex in self._examples]
        unique_texts = ExactDeduplicator().filter_duplicates(texts)
        text_set = set(unique_texts)
        deduped = [ex for ex in self._examples if str(ex.get("text", "")) in text_set]
        self._examples.clear()
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
    ) -> None:
        self.orchestrator = orchestrator
        self.buffer = buffer or FeedbackBuffer(min_examples=min_examples)

    def submit(
        self,
        artifact: Mapping[str, Any],
        *,
        train_step: Callable[[int, Any], Dict[str, float]],
        min_examples: Optional[int] = None,
    ) -> Optional[TrainingContext]:
        self.buffer.add(artifact)
        if not self.buffer.ready(min_examples):
            return None
        batches = self.buffer.as_batches()
        return self.orchestrator.fit(batches, train_step=train_step)

    @classmethod
    def create_default(
        cls,
        config: Mapping[str, Any],
        *,
        seed: int = 1337,
        max_steps: int = 100,
        min_examples: int = 8,
    ) -> FeedbackLoopTrainer:
        repro = ReproducibilityController(seed=seed)
        repro.set_seeds()
        orch = TrainingOrchestrator(config, reproducibility=repro, max_steps=max_steps)
        return cls(orch, min_examples=min_examples)


def corrections_to_manifest(examples: List[Mapping[str, Any]], output_dir: str) -> Any:
    """Write flushed corrections to JSONL and build a manifest."""
    path = Path(output_dir) / "corrections.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(dict(ex)) + "\n")
    manifest = build_dataset_manifest(path, dataset_id="agent-corrections")
    return provenance_from_manifest(manifest)
