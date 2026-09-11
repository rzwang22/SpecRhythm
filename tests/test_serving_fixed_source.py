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
