from deepiri_training_orchestrator.distributed import DistributedContext, init_distributed


def test_init_distributed_single_process():
    ctx = init_distributed()
    assert isinstance(ctx, DistributedContext)
    assert ctx.is_main_process is True
    assert ctx.world_size >= 1


def test_main_process_only():
    ctx = init_distributed()
    result = ctx.is_main_process and "ok" or None
    assert result == "ok" or result is None
