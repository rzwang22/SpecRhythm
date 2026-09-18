"""Run the actual GPU-check coordination with deterministic device operations only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_vllm_draft import FakeWorker, next_token

from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
from specrhythm.continuation.gpu_check import (
    errors,
    main,
    run_checks,
    status,
    stop,
    supervise,
)
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.s2_pool import publish


def physical(backend):
    for state in backend.states.values():
        memory = backend.worker.memory[state.internal_id]
        assert tuple(memory[:len(state.prefix)]) == state.prefix
        assert state.materialized <= len(memory)
        assert state.next_logits == next_token(memory[:state.materialized])
    return {"physical_cpu_array_checked": True, "live_requests": len(backend.states)}


def backend(worker=None):
    return RollingVllmDraftBackend(
        SimpleNamespace(max_model_len=4096), worker=worker or FakeWorker()
    )


def test_real_backend_and_core_coordinate_both_receipt_orders_and_repair():
    device = backend()
    result = run_checks(device, [
        {"request_id": "one", "prefix": (1, 2, 3), "remaining_output_tokens": 64},
        {"request_id": "two", "prefix": (4, 5, 6), "remaining_output_tokens": 64},
    ], eos_token_ids=(), vocab_size=100, snapshot=physical)
    assert result["constructed_cases"]["success"]["count"] == 10
    assert result["constructed_cases"]["parent_rejected"]["count"] == 2
    assert result["constructed_cases"]["bridge_mismatch"]["count"] == 2
    assert {r["target_first"] for r in result["events"] if "target_first" in r} == {False, True}
    assert all(r["exact"] for r in result["reference_comparisons"])
    assert result["live_request_initial_prefills"] == 2
    assert all(r["recovery_jobs"] == 2 for r in result["requests"].values())
    assert not device.states and not device._gpu_continuations
    assert result["performance_result"] is False
    assert result["target_observed_events"].startswith("NOT_OBSERVED")
    assert result["gpu_overlap"].startswith("UNKNOWN")
    device.shutdown()


def test_three_request_physical_check_exercises_mixed_batch_settlement():
    device = backend()
    result = run_checks(device, [
        {'request_id': f'r-{i}', 'prefix': (10, 20+i), 'remaining_output_tokens': 64}
        for i in range(3)
    ], eos_token_ids=(), vocab_size=100, snapshot=physical)
    mixed = [r for r in result['events'] if r.get('stage') == 2]
    assert {r['construction'] for r in mixed} == {
        'success', 'parent_rejected', 'bridge_mismatch'}
    assert all(len(r['settlement_batch_request_ids']) == 3 for r in mixed)
    assert len(result['reference_comparisons']) == 21
    assert all(r['exact'] for r in result['reference_comparisons'])
    assert device.metrics.batches['commit'][2] > 0
    assert device.metrics.batches['eager'][3] > 0
    assert not device.states and not device._gpu_continuations
    device.shutdown()


@pytest.mark.parametrize("remaining", [1, 2, 5, 6, 9])
def test_real_check_keeps_legal_length_and_tail_without_forcing_k4(remaining):
    device = backend()
    result = run_checks(device, [
        {"request_id": "one", "prefix": (1, 2, 3), "remaining_output_tokens": remaining},
    ], eos_token_ids=(), vocab_size=100, snapshot=physical)
    assert result["requests"]["one"]["committed_tokens"] == remaining
    assert not device.states
    device.shutdown()


def test_natural_eos_does_not_manufacture_missing_construction_events():
    device = backend()
    eos = (next_token((1, 2, 3)),)
    result = run_checks(device, [
        {"request_id": "one", "prefix": (1, 2, 3), "remaining_output_tokens": 64},
    ], eos_token_ids=eos, vocab_size=100, snapshot=physical)
    assert result["requests"]["one"]["committed_tokens"] == 1
    assert all(v["status"] == "NOT_OBSERVED" for v in result["constructed_cases"].values())
    assert not device.states
    device.shutdown()


def test_reference_comparison_detects_incorrect_eager_device_logits():
    class Damaged(FakeWorker):
        def materialize(self, rows, purpose):
            result = super().materialize(rows, purpose)
            return ({key: value + 1 for key, value in result.items()}
                    if purpose == "eager" else result)

    device = backend(Damaged())
    try:
        with pytest.raises(DataError, match="differs from ordinary reference"):
            run_checks(device, [{"request_id": "r", "prefix": (1, 2, 3),
                                 "remaining_output_tokens": 64}],
                       eos_token_ids=(), vocab_size=100, snapshot=lambda b: {})
    finally:
        device.shutdown()


def test_read_only_commands_do_not_construct_any_gpu_backend(tmp_path, monkeypatch, capsys):
    import specrhythm.continuation.gpu_backend as gpu

    monkeypatch.setattr(gpu, "RollingVllmDraftBackend", lambda *a, **k: pytest.fail("GPU import"))
    assert main(["status", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"state": None, "result": None}
    assert main(["errors", "--root", str(tmp_path)]) == 0
    assert status(tmp_path)["state"] is None
    assert all(v is None for v in errors(tmp_path).values())


def test_supervisor_success_retains_owned_identity_and_real_exit(tmp_path):
    script = ("import json,pathlib; "
              "pathlib.Path('result.json').write_text(json.dumps({'valid':True}))")
    # A real CPU child proves process ownership/exit handling without invoking the GPU worker.
    command = [sys.executable, "-c", f"import os; os.chdir({str(tmp_path)!r}); {script}"]
    supervise(tmp_path, command, os.environ.copy(), timeout=10, drain_timeout=2)
    state = read_json(tmp_path / "state.json")
    assert state["status"] == "COMPLETE" and state["exit_code"] == 0
    assert state["remaining_owned_pids"] == [] and state["owner_identity"]


def test_supervisor_tracks_descendant_after_worker_leader_exits(tmp_path):
    script = (
        "import pathlib,subprocess,sys,time; "
        f"root=pathlib.Path({str(tmp_path)!r}); "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.4)']); "
        "(root/'result.json').write_text('{\"valid\":true}'); time.sleep(0.05)"
    )
    supervise(tmp_path, [sys.executable, "-c", script], os.environ.copy(),
              timeout=10, drain_timeout=3)
    state = read_json(tmp_path / "state.json")
    assert state["status"] == "COMPLETE" and state["remaining_owned_pids"] == []
    assert len(state["owner_identity"]) == 2 and state["ownership_verified"]


def test_controlled_stop_reaps_only_recorded_worker_and_preserves_failure(tmp_path):
    directory = tmp_path / "rolling-eager-gpu-check"
    directory.mkdir()
    publish(directory / "state.json", {"status": "RUNNING"})
    assert stop(tmp_path)["action"] == "controlled stop requested"
    with pytest.raises(DataError, match="check failed"):
        supervise(directory, [sys.executable, "-c", "import time; time.sleep(20)"],
                  os.environ.copy(), timeout=10, drain_timeout=2)
    state = status(tmp_path)["state"]
    assert state["status"] == "FAILED" and state["remaining_owned_pids"] == []
    assert errors(tmp_path)["diagnostic-primary-error.json"]["error"] == "operator stop"
    assert any(a["signal"] == "SIGTERM" for a in state["actions"])


def test_unavailable_process_ownership_fails_before_starting_child(tmp_path, monkeypatch):
    import specrhythm.phase4.owned_processes as processes

    def denied():
        raise PermissionError("process inspection unavailable")

    monkeypatch.setattr(processes, "process_table", denied)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("child spawned"))
    with pytest.raises(DataError, match="check failed"):
        supervise(tmp_path, [sys.executable, "-c", "pass"], {}, timeout=1, drain_timeout=1)
    assert read_json(tmp_path / "state.json")["exit_code"] is None
    assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == (
        "process inspection unavailable"
    )


def test_lost_process_inspection_preserves_first_error_and_failed_cleanup(tmp_path, monkeypatch):
    import specrhythm.phase4.owned_processes as processes

    actual, calls = processes.process_table, []

    def fails_after_launch():
        calls.append(True)
        if len(calls) > 2:
            raise PermissionError("process inspection lost")
        return actual()

    monkeypatch.setattr(processes, "process_table", fails_after_launch)
    with pytest.raises(DataError, match="check failed"):
        supervise(tmp_path, [sys.executable, "-c", "import time; time.sleep(20)"],
                  os.environ.copy(), timeout=5, drain_timeout=2)
    state = read_json(tmp_path / "state.json")
    assert state["status"] == "FAILED" and state["exit_code"] == -15
    assert state["cleanup_status"] == "FAILED" and not state["ownership_verified"]
    assert read_json(tmp_path / "diagnostic-primary-error.json")["error"] == (
        "process inspection lost"
    )


def test_module_help_is_gpu_free():
    result = subprocess.run([sys.executable, "-m", "specrhythm.continuation.gpu_check", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "inspection never initializes CUDA" in result.stdout


def test_existing_verify_hook_patch_is_before_actual_forward():
    patch = Path("integrations/vllm/patches/0001-custom-proposer-request-and-verify-hooks.patch")
    source = patch.read_text()
    assert "@@ -4334,0 +4335,15" in source
    assert source.index('"on_target_verify_start"') < source.index('"on_target_verify_end"')
    # Insertion into the pinned function must remain upstream of the ordinary
    # forward boundary, as evidenced by the retained source audit in the design.
    source_map = Path("src/specrhythm/phase4/vllm_draft_api.json").read_text()
    assert "gpu_model_runner.py" in source_map
