"""Pinned source checks for the real short-window cancellation and event hooks."""

from test_serving_s2_source import method
from test_serving_s2_source import source as _source

source = _source


def test_real_engine_abort_reaches_allocator_and_does_not_decode(source):
    engine = method(source / "v1/engine/llm_engine.py", "LLMEngine", "abort_request")
    assert "output_processor.abort_requests" in engine
    assert "engine_core.abort_requests" in engine
    scheduler = method(source / "v1/core/sched/scheduler.py", "Scheduler", "finish_requests")
    assert "self._free_request" in scheduler and "execute_model" not in scheduler
    free = method(source / "v1/core/sched/scheduler.py", "Scheduler", "_free_request")
    assert "self._free_blocks(request)" in free
    blocks = method(source / "v1/core/sched/scheduler.py", "Scheduler", "_free_blocks")
    assert "del self.requests[request.request_id]" in blocks
    deferred = method(source / "v1/core/sched/scheduler.py", "Scheduler", "_free_request_blocks")
    assert "request.last_sched_seq <= self.processed_step_seq" in deferred
    assert "kv_cache_manager.free(request)" in deferred


def test_final_collective_rpc_supports_real_bounded_timeout(source):
    body = method(source / "entrypoints/llm.py", "LLM", "collective_rpc")
    assert "timeout: float | None" in body
    assert "self.llm_engine.collective_rpc(method, timeout, args, kwargs)" in body


def test_worker_model_and_current_request_rows_are_actual_mrv1_contract(source):
    body = method(source / "v1/worker/gpu_model_runner.py", "GPUModelRunner", "get_model")
    assert "self.model" in body
    update = method(source / "v1/worker/gpu_model_runner.py", "GPUModelRunner", "_update_states")
    assert "self.input_batch.remove_request(req_id)" in update


def test_scan_stop_occurs_before_model_dispatch_and_abort_uses_non_deferred_free(source):
    step = method(source / "v1/engine/core.py", "EngineCore", "step")
    assert step.index("self.scheduler.schedule(") < step.index(
        "self.model_executor.execute_model")
    init = method(source / "v1/core/sched/scheduler.py", "Scheduler", "__init__")
    assert "self.defer_block_free = False" in init
    free = method(source / "v1/core/sched/scheduler.py", "Scheduler", "_free_request_blocks")
    assert "if not self.defer_block_free or" in free
    assert "self.kv_cache_manager.free(request)" in free


def test_inproc_wait_unwinds_before_post_step_and_output_mutation(source):
    import ast
    from types import SimpleNamespace

    import pytest

    from specrhythm.serving.decode_scan_readiness import ScanBatchWait

    # Execute the pinned InprocClient get_output body, not a hand-written approximation.
    text = method(source / "v1/engine/core_client.py", "InprocClient", "get_output")
    tree = ast.parse("from __future__ import annotations\n" + text)
    scope = {}
    exec(compile(tree, "pinned-get_output", "exec"), scope)
    after = []

    def wait():
        raise ScanBatchWait({"reason": "draft_inflight"})

    owner = SimpleNamespace(engine_core=SimpleNamespace(
        step_fn=wait, post_step=lambda **kw: after.append(kw)))
    with pytest.raises(ScanBatchWait):
        scope["get_output"](owner)
    assert not after
    llm = method(source / "v1/engine/llm_engine.py", "LLMEngine", "step")
    assert llm.index("self.engine_core.get_output()") < llm.index(
        "self.output_processor.process_outputs(")
