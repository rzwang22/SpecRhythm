"""Read-only pinned-vLLM source contracts for the new S2 control boundary."""

import ast
import os
from pathlib import Path

import pytest


@pytest.fixture
def source():
    path = os.environ.get("SR_PHASE4B3_SOURCE_AUDIT")
    if not path:
        pytest.skip("pinned vLLM source contract runs in Linux CI")
    return Path(path) / "vllm"


def method(path, cls, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, function)


def test_inproc_target_step_has_no_autonomous_busy_loop(source):
    body = method(source / "v1/engine/core_client.py", "InprocClient", "get_output")
    assert "self.engine_core.step_fn()" in body
    assert "self.engine_core.post_step(model_executed=model_executed)" in body
    assert not any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(ast.parse(body)))
    entry = method(source / "v1/engine/llm_engine.py", "LLMEngine", "from_engine_args")
    assert "envs.VLLM_ENABLE_V1_MULTIPROCESSING" in entry


def test_unscheduled_input_rows_keep_private_cached_KV_until_explicit_finish(source):
    update = method(source / "v1/worker/gpu_model_runner.py", "GPUModelRunner", "_update_states")
    removal = update.index("for req_id in unscheduled_req_ids:")
    after = update[removal : update.index("is_ngram_gpu", removal)]
    assert "self.input_batch.remove_request(req_id)" in after
    assert "self.requests.pop" not in after and "free(" not in after
    assert "self.requests.pop(req_id, None)" in update  # explicit finished_req_ids path
    kv = method(source / "v1/core/kv_cache_manager.py", "KVCacheManager", "get_block_ids")
    assert "self.get_blocks(request_id).get_block_ids()" in kv
    getter = method(source / "v1/core/kv_cache_manager.py", "KVCacheManager", "get_blocks")
    assert "self.coordinator.get_blocks(request_id)" in getter


def test_stock_scheduler_preemption_and_block_ownership_are_exposed(source):
    schedule = method(source / "v1/core/sched/scheduler.py", "Scheduler", "schedule")
    assert "self._preempt_request(" in schedule
    assert "preempted_req_ids=self.reset_preempted_req_ids" in schedule
    assert "scheduled_spec_decode_tokens[request.request_id] = spec_token_ids" in schedule
    assert "request.spec_token_ids = []" in schedule
