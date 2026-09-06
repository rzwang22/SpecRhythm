"""Run with SR_PHASE4B3_SOURCE_AUDIT pointing at an exact CPU source export."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from specrhythm.phase4.vllm_draft_worker import API_PATH, EXECUTE_FIELDS, audit_source_files


@pytest.fixture(scope="module")
def source():
    path = os.environ.get("SR_PHASE4B3_SOURCE_AUDIT")
    if not path:
        pytest.skip("source-fixture tests run in the dedicated pinned-vLLM Linux CI job")
    root = Path(path)
    audit_source_files(root, patched=False)
    return root


def test_exact_signatures_and_state_fields(source):
    for entry in json.loads(API_PATH.read_text())["files"]:
        tree = ast.parse((source / entry["path"]).read_text())
        top = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for symbol in entry["symbols"]:
            node = top[symbol["symbol"]]
            if isinstance(node, ast.FunctionDef):
                assert ast.unparse(node.args) == symbol["signature"]
            else:
                fields = [
                    n.target.id
                    for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                ]
                assert fields == symbol["fields"]
                methods = {n.lineno: n for n in node.body if isinstance(n, ast.FunctionDef)}
                for method in symbol["methods"]:
                    assert ast.unparse(methods[method["line"]].args) == method["signature"]
                if node.name == "ExecuteModelState":
                    assert tuple(fields) == EXECUTE_FIELDS


def test_old_classes_absent_and_rebase_path_present(source):
    old = {
        "SpecDecodeWorker",
        "MultiStepWorker",
        "TP1DraftModelRunner",
        "SmallerTpProposerWorker",
        "Top1Proposer",
        "SpeculativeProposals",
    }
    for path in (source / "vllm").rglob("*.py"):
        text = path.read_text()
        assert not any(f"class {name}(" in text or f"class {name}:" in text for name in old)
    runner = (source / "vllm/v1/worker/gpu_model_runner.py").read_text()
    assert "self._update_streaming_request(req_id, new_req_data)" in runner
    assert "req_state.num_computed_tokens = new_req_data.num_computed_tokens" in runner
    assert "req_state.output_token_ids.clear()" in runner
    assert "State error: sample_tokens() must be called" in runner


def test_worker_builds_real_scheduler_metadata_for_ragged_rebases(source, monkeypatch):
    """Run adapter glue with the pinned real SchedulerOutput classes, CPU allocator/RPC."""
    from specrhythm.phase4.draft_batch import DraftMaterialization
    from specrhythm.phase4.draft_metrics import DraftMetrics
    from specrhythm.phase4.vllm_draft_worker import VllmDraftWorker

    module_name = "vllm.v1.core.sched.output"
    output_module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, output_module)
    source_code = (source / "vllm/v1/core/sched/output.py").read_text()
    exec(compile(source_code, "pinned-output.py", "exec"), output_module.__dict__)

    class Request:
        def __init__(self, request_id, prompt_token_ids, sampling_params, pooling_params):
            self.request_id, self.prompt_token_ids = request_id, prompt_token_ids
            assert pooling_params is None

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(Request=Request, RequestStatus=SimpleNamespace(RUNNING="running")),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)),
    )

    class Allocator:
        def __init__(self):
            self.blocks, self.steps, self.allocations = {}, 0, []

        def new_step_starts(self):
            self.steps += 1

        def allocate_slots(self, view, *, num_new_tokens):
            self.allocations.append((view.request_id, view.num_computed_tokens, num_new_tokens))
            needed = (view.num_computed_tokens + num_new_tokens + 1) // 2
            blocks = self.blocks.setdefault(view.request_id, [])
            additional = list(range(len(blocks), needed))
            blocks.extend(additional)
            return SimpleNamespace(get_block_ids=lambda: (additional,))

        def get_block_ids(self, rid):
            return (list(self.blocks[rid]),)

        def free(self, view):
            del self.blocks[view.request_id]

    worker = VllmDraftWorker.__new__(VllmDraftWorker)
    worker.owner_thread, worker.closed = threading.get_ident(), False
    worker.metrics, worker.kv = DraftMetrics(), Allocator()
    worker.blocks_allocated = worker.blocks_freed = worker.blocks_peak = worker.batch_sequence = 0
    worker.views = {}
    worker.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=16)
    )
    worker.cache_config = SimpleNamespace(kv_cache_groups=[1], needs_kv_cache_zeroing=False)
    worker.config_context = lambda cfg: nullcontext()
    outputs = []

    def rpc(name, *, args, single_value):
        assert name == "sr_draft_materialize" and single_value is True
        output = args[0]
        outputs.append(output)
        assert output.scheduled_cached_reqs.req_ids == []
        assert output.scheduled_spec_decode_tokens == {}
        if output.total_num_scheduled_tokens:
            assert not output.finished_req_ids
            worker.metrics.forward(
                worker.phase, len(output.num_scheduled_tokens), output.total_num_scheduled_tokens
            )
        return {r.req_id: r.prompt_token_ids[-1] for r in reversed(output.scheduled_new_reqs)}

    worker.executor = SimpleNamespace(collective_rpc=rpc)
    worker.fence = lambda reason: None
    worker.materialize(
        [DraftMaterialization("b", (1, 2, 3), 0), DraftMaterialization("a", (9, 8), 0)], "setup"
    )
    original_blocks = worker.kv.get_block_ids("b")
    result = worker.materialize(
        [DraftMaterialization("a", (9, 8, 7, 6), 2), DraftMaterialization("b", (1, 5), 1)],
        "commit",
    )
    assert result == {"a": 6, "b": 5}
    assert outputs[-1].num_scheduled_tokens == {"a": 2, "b": 1}
    assert outputs[-1].num_common_prefix_blocks == [0]
    assert worker.kv.get_block_ids("b") == original_blocks  # Retain allocation; mask stale tail.
    assert worker.metrics.forwards == {"setup": 1, "commit": 1}
    assert worker.kv.steps == 2 and len(outputs) == 2
    worker.release(["a", "b"])
    assert outputs[-1].finished_req_ids == {"a", "b"}
    assert worker.blocks_allocated == worker.blocks_freed == 4
    assert worker.views == {}


def test_existing_five_patch_fixture_and_drift_rejection(source, tmp_path):
    # Only an isolated source fixture is patched, never the source/installed runtime.
    for entry in json.loads(API_PATH.read_text())["files"]:
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / entry["path"], target)
    patches = Path("integrations/vllm/patches")
    names = [
        "0001-custom-proposer-request-and-verify-hooks.patch",
        "0002-scheduler-request-admissibility-hook.patch",
        "0003-target-forward-timing-observer.patch",
        "0004-gate3-numerical-observer.patch",
        "0005-dual-sampled-row-context.patch",
    ]
    for name in names:
        subprocess.run(
            [
                "patch",
                "--batch",
                "-p1",
                "-d",
                str(tmp_path),
                "-i",
                str((patches / name).resolve()),
            ],
            check=True,
            capture_output=True,
        )
    evidence = audit_source_files(tmp_path)
    assert len(evidence["files"]) == len(json.loads(API_PATH.read_text())["files"])
    runner = tmp_path / "vllm/v1/worker/gpu_model_runner.py"
    runner.write_text(runner.read_text() + "\n# source drift\n")
    with pytest.raises(RuntimeError, match="source mismatch"):
        audit_source_files(tmp_path)
