from deepiri_training_orchestrator import (
    ReproducibilityController,
    initialize_deterministic_training,
)


def test_fingerprint_stable() -> None:
    r = ReproducibilityController(seed=1)
    fp1 = r.generate_training_fingerprint({"a": 1})
    fp2 = r.generate_training_fingerprint({"a": 1})
    assert fp1 == fp2


def test_fingerprint_changes_with_config() -> None:
    r = ReproducibilityController(seed=1)
    fp1 = r.generate_training_fingerprint({"a": 1})
    fp2 = r.generate_training_fingerprint({"a": 2})
    assert fp1 != fp2


def test_initialize_deterministic_training() -> None:
    c = initialize_deterministic_training(seed=42)
    assert c.seed == 42


def test_worker_init_fn_runs() -> None:
    r = ReproducibilityController(seed=7)
    fn = r.get_dataloader_worker_init_fn()
    fn(0)
    fn(1)
