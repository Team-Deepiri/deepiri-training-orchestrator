"""
Deterministic training and reproducibility: global seeds, CUDA determinism, config fingerprinting.
Extracted from Helox ``core/reproducibility_controller.py``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


class ReproducibilityController:
    """
    Controls sources of randomness for reproducible training (Python, NumPy, PyTorch, CUDA).
    """

    def __init__(
        self,
        seed: int = 1337,
        deterministic: bool = True,
        benchmark: bool = False,
    ) -> None:
        self.seed = seed
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.fingerprint: Optional[str] = None

    def set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            cudnn.deterministic = True
            cudnn.benchmark = self.benchmark

        logger.info("Seeds set to %s (deterministic=%s)", self.seed, self.deterministic)

    def get_dataloader_worker_init_fn(self):
        """Return a DataLoader ``worker_init_fn`` for deterministic multi-worker loading."""

        def worker_init_fn(worker_id: int) -> None:
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return worker_init_fn

    def generate_training_fingerprint(
        self,
        config: Dict[str, Any],
        code_hash: Optional[str] = None,
    ) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        fingerprint_data = {
            "seed": self.seed,
            "config": config_str,
            "code_hash": code_hash or "unknown",
        }
        fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(fingerprint_json.encode()).hexdigest()[:16]
        self.fingerprint = fingerprint
        logger.info("Training fingerprint: %s", fingerprint)
        return fingerprint

    def save_fingerprint(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_data = {
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "deterministic": self.deterministic,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint_data, f, indent=2)
        logger.info("Fingerprint saved to %s", output_path)

    def verify_reproducibility(
        self,
        checkpoint_path: Path,
        expected_fingerprint: Optional[str] = None,
    ) -> bool:
        fingerprint_file = Path(checkpoint_path) / "training_fingerprint.json"
        if not fingerprint_file.exists():
            logger.warning("Fingerprint file not found: %s", fingerprint_file)
            return False

        with open(fingerprint_file, encoding="utf-8") as f:
            saved_data = json.load(f)

        saved_fingerprint = saved_data.get("fingerprint")
        if expected_fingerprint:
            matches = saved_fingerprint == expected_fingerprint
            if not matches:
                logger.error(
                    "Fingerprint mismatch: expected %s, got %s",
                    expected_fingerprint,
                    saved_fingerprint,
                )
            return matches
        return True


def initialize_deterministic_training(
    seed: int = 1337,
    deterministic: bool = True,
) -> ReproducibilityController:
    controller = ReproducibilityController(seed=seed, deterministic=deterministic)
    controller.set_seeds()
    return controller
