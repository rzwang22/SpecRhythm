"""Explicit eager CLI/config/capacity/result integration, with CPU device substitutes."""

import copy
import csv
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_serial import phase4_config as _config
from test_serving_decode_scan_cli import report
from test_serving_decode_scan_results import evidence
from test_serving_fixed import clock_fixture
from test_serving_s1 import execution, freeze_serial_inputs
from test_serving_s2 import profile, ranks

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving import decode_scan_cli, fixed_cli, fixed_runtime, s2_runtime
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.decode_scan_plan import (
    BATCHES,
    MODES,
    SCHEMA,
    manifest,
    options,
    selected_point,
)
from specrhythm.serving.decode_scan_results import timing
from specrhythm.serving.eager_results import (
    COUNTERS,
    eager_columns,
    eager_overlap,
    summarize_eager,
)
from specrhythm.serving.fixed_plan import capacity_metadata, point
from specrhythm.serving.runtime_profile import PROFILE_ENV
from specrhythm.serving.s2_cli import child_command
from specrhythm.serving.s2_plan import capacity_for, sealed
from specrhythm.serving.s2_pool import publish

phase4_config = _config


@pytest.fixture
def scan_root(tmp_path):
    config = sealed(dict(
        schema_version=SCHEMA, execution={"git_commit": "a" * 40}, workload_sha256="work",
        pool_size=360, options=options(),
        points=[selected_point(mode, batch) for batch in BATCHES for mode in MODES],
        optional_points=[selected_point("serial-eager", batch) for batch in BATCHES],
    ))
    publish(tmp_path / "scan-config.json", config)
    return tmp_path


def test_explicit_mode_preserves_unsplit_shape_and_baseline_defaults():
    assert MODES == ("target", "serial", "pingpong")
    assert "serial-eager" not in fixed_cli.MODES
    selected = selected_point("serial-eager", 16)
    assert selected["runtime_mode"] == "serial-eager"
    assert point("serial-eager")["runtime_mode"] == "serial-eager"
    args = fixed_cli.parser().parse_args(
        ["short", "--root", "/tmp/test", "--mode", "serial-eager"])
    assert args.mode == "serial-eager"
    cap = capacity_metadata("serial-eager", active_limit=16)
    assert cap["cohort_count"] == 0 and cap["per_cohort_capacity"] is None
    assert cap["max_requests_per_target_forward"] == 16
    assert cap["draft_speculative_capacity_tokens"] == 9
    assert cap["draft_extra_speculative_tokens"] == 5
    assert cap["proposal_budget"] == cap["eager_candidate_length"] == 4
    frozen = manifest({}, [str(i) for i in range(360)], "work", options(), 16)
    assert frozen["fixed_diagnostic"]["capacity"]["serial-eager"]["active_request_limit"] == 16
    assert frozen["fixed_diagnostic"]["options"]["samples"] is None


def test_explicit_eager_cli_runs_only_one_frozen_optional_point(scan_root, monkeypatch):
    calls = []
    monkeypatch.setattr(decode_scan_cli, "point_manifest", lambda root, b: root / f"B{b}.json")

    def run(root, selected, *, probe, manifest_path):
        calls.append((selected, probe, manifest_path))
        return report(root, selected, probe=probe)

    monkeypatch.setattr(decode_scan_cli, "run_point", run)
    assert decode_scan_cli.main([
        "capacity", "--root", str(scan_root), "--mode", "serial-eager", "--batch", "16",
    ]) == 0
    assert calls[0][0]["mode"] == "serial-eager" and calls[0][1]
    assert decode_scan_cli.main([
        "run", "--root", str(scan_root), "--mode", "serial-eager", "--batch", "16",
        "--single-point",
    ]) == 0
    assert len(calls) == 2
    assert calls[1][0]["test_order"] == "independent-single-point"
    decode_scan_cli.run(scan_root, batch=16)
    assert [row[0]["mode"] for row in calls[2:]] == list(MODES)
    summary = decode_scan_cli.summary(scan_root)
    eager = [row for row in summary["points"] if row["mode"] == "serial-eager"]
    assert len(eager) == 1 and eager[0]["batch"] == 16 and eager[0]["sub_batch"] is None
    assert all(row["mode"] != "serial-eager" or row["batch"] == 16
               for row in summary["points"])
    cmd = child_command("child", scan_root, "serial-eager", scan_root / "B16.json",
                        scan_root / "run", diagnostic=True)
    assert cmd[cmd.index("-m") + 1] == "specrhythm.serving.fixed_cli"
    assert cmd[cmd.index("--mode") + 1] == "serial-eager"


def test_engine_routing_selects_eager_proposer_with_full_serial_batch(phase4_config, monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **kw: calls.append(kw)))
    s2_runtime.make_engine(phase4_config, "serial-eager", classes=fixed_runtime.CLASSES,
                           sequence_limit=512, query_limit=4096)
    config = calls[0]
    assert config["scheduler_cls"] == "specrhythm.serving.fixed_scheduler.FixedSerialScheduler"
    assert config["speculative_config"] == {
        "model": "specrhythm.serving.eager_proposer.EagerSerialProposer",
        "method": "custom_class", "num_speculative_tokens": 4,
    }
    assert config["max_num_seqs"] == 512 and config["max_num_batched_tokens"] == 4096
    assert not config["async_scheduling"]
    assert fixed_runtime.CLASSES["serial"][1].endswith("S2SerialProposer")


def test_actual_configuration_uses_serial_setup_and_no_dual_runtime(
    tmp_path, monkeypatch, phase4_config
):
    old, previous, _ = execution(tmp_path / "s1")
    freeze_serial_inputs(tmp_path, old, previous)
    path, _, _ = profile(tmp_path / "s2", execution=previous["execution"])
    directory = tmp_path / "run"
    directory.mkdir()
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv(PROFILE_ENV, str(path))
    monkeypatch.setenv("SR_S2_DRAFT_SOCKET", str(tmp_path / "draft.sock"))
    monkeypatch.setattr(s2_runtime, "validate_installed_patch_stack", lambda *a: {})
    config, _, rows = s2_runtime.configure(tmp_path, path, directory, "serial-eager")
    assert rows and config.proposal_budget == 4
    assert os.environ["SR_PHASE4_RESIDENT_CONSUMER"] == "serial"
    assert os.environ["SR_PHASE4_DUAL_BATCH"] == "0"
    assert os.environ["SR_PHASE4_DUAL_RESIDENT"] == "0"
    setup = read_json(directory / "setup-control.json")
    assert setup["consumer"] == "serial"


def test_explicit_service_factory_adds_physical_backend_to_provided_fixed_class(
    tmp_path, monkeypatch
):
    from specrhythm.continuation.gpu_backend import GPUContinuationBackendMixin
    from specrhythm.serving import eager_draft

    constructed = []

    class ProvidedBackend:
        def __init__(self, config):
            self.metrics = SimpleNamespace(batches={})
            self._provenance = {}
            self.provenance = self._provenance
            constructed.append(self)

    def capture_owner(factory):
        factory()
        raise RuntimeError("factory captured without starting GPU")

    monkeypatch.setattr(eager_draft, "control", lambda: {"requests": {"r": {}}})
    monkeypatch.setattr(eager_draft, "EagerOwner", capture_owner)
    monkeypatch.setattr(eager_draft, "EagerSerialMachine", lambda backend, **kw: backend)
    with pytest.raises(RuntimeError, match="factory captured"):
        eager_draft.serve(object(), tmp_path, tmp_path / "draft.sock",
                          backend_class=ProvidedBackend)
    assert isinstance(constructed[0], GPUContinuationBackendMixin)
    assert isinstance(constructed[0], ProvidedBackend)
    assert "eager" in constructed[0].metrics.batches
    assert read_json(tmp_path / "draft-startup.json")["rolling_eager_gpu"][
        "incremental_batched_steps"]


def test_capacity_reserves_extra_five_tokens_only_for_eager_draft():
    rows = [SimpleNamespace(prompt_length=16, maximum_new_tokens=12) for _ in range(360)]
    draft = {**ranks()[0], "mode": "serial-eager"}
    baseline = capacity_for(rows, draft, active_limit=16)
    eager = capacity_for(rows, draft, active_limit=16, speculative_tokens=9)
    assert eager["required_blocks"] == baseline["required_blocks"] + 16
    assert eager["extra_speculative_tokens"] == 5
    assert eager["prefix_blocks"] == baseline["prefix_blocks"]
    target = {**draft, "role": "target"}
    assert "extra_speculative_tokens" not in capacity_for(rows, target, active_limit=16)
    available = baseline["required_blocks"]
    # Preserve the measured reserve while choosing a small synthetic block pool.
    tight = {**draft, "num_gpu_blocks": available}
    normal = capacity_for(rows, tight, active_limit=16)
    extra = capacity_for(rows, tight, active_limit=16, speculative_tokens=9)
    assert extra["required_blocks"] > normal["required_blocks"]
    with pytest.raises(DataError, match="baseline K4"):
        capacity_for(rows, draft, speculative_tokens=3)


def test_real_initial_work_route_reuses_serial_bulk_payload():
    clock, ids = clock_fixture()
    warm = {rid: SimpleNamespace(logical_committed_prefix_count=7,
                                logical_committed_prefix_sha256="frozen") for rid in ids[:16]}
    packet = {"eos_token_ids": [999], "initial_proposals": {}}
    calls = []

    def call(operation, payload):
        calls.append((operation, payload))
        return {"proposals": [{"request_id": row["request_id"]} for row in payload["proposals"]],
                "service_send_ns": 10, "transport_end_ns": 11}

    s2_runtime.initial_work("serial-eager", ids[:16], clock, warm,
                            SimpleNamespace(call=call), packet)
    assert len(calls) == 1 and calls[0][0] == "synchronize_and_batch_propose"
    assert len(calls[0][1]["proposals"]) == len(packet["initial_proposals"]) == 16
    assert not calls[0][1]["synchronizations"]


def test_natural_finish_sends_authoritative_eager_prefix_before_release():
    clock, ids = clock_fixture()
    rid = clock.admit(100)[0]
    tokens = [10, 20, 21, 999]
    calls = []

    def call(operation, payload):
        assert not clock.rows[rid]["resources_released"]
        calls.append((operation, payload))
        return {}

    output = SimpleNamespace(request_id=rid, finished=True,
                             outputs=[SimpleNamespace(token_ids=tokens, finish_reason="stop")])
    packet = {"initial_proposals": {}, "initial_enqueues": {}, "eos_token_ids": [999]}
    fixed_runtime.commit_outputs(
        clock, [output], packet, SimpleNamespace(call=call), "serial-eager")
    expected = list(clock.definitions[rid].prompt_token_ids) + tokens
    assert calls == [("finish_request", {"request_id": rid, "committed_prefix": expected,
                                         "committed_prefix_hash": token_prefix_hash(expected),
                                         "terminal": True, "eos_token_ids": [999]})]
    assert clock.rows[rid]["resources_released"]


def eager_evidence():
    rounds = dict(admissions=1, started=1, completed=1, parent_full_accepts=1,
                  bridge_matches=1, promotions=1, early_generated_tokens=5,
                  bridge_generated_tokens=1, draft_materialized_tokens=5,
                  committed_tokens=5, parent_accepted_tokens=4, bonus_tokens=1)
    events = [
        {"phase": "warmup", "start_ns": 10, "end_ns": 50, "counter_delta": rounds},
        {"phase": "round", "start_ns": 130, "end_ns": 150, "counter_delta": {
            **rounds, "verified_promoted_candidates": 4, "accepted_promoted_candidates": 4}},
        {"phase": "wait", "start_ns": 90, "end_ns": 120,
         "counter_delta": {"unhidden_wait_ns": 30}},
        {"phase": "drain", "start_ns": 210, "end_ns": 220,
         "counter_delta": {"discarded_early_tokens": 5}},
    ]
    counters = dict.fromkeys(COUNTERS, 0)
    for event in events:
        for name, count in event["counter_delta"].items():
            counters[name] += count
    return {"rolling_eager": {"schema_version": "specrhythm.rolling-eager.v1",
                              "events": events, "counters": counters,
                              "pending_work": [], "owner_stopped": True}}, {
        "measurement_start_ns": 100, "measurement_end_ns": 200,
    }


def test_event_window_separates_generated_promoted_verified_and_accepted_work():
    backend, runtime = eager_evidence()
    value = summarize_eager(backend, runtime)
    assert value["lifetime_counters"]["started"] == 2
    assert value["window_counters"]["started"] == 1
    assert value["window_counters"]["early_generated_tokens"] == 5
    assert value["window_counters"]["promotions"] == 1
    assert value["window_counters"]["verified_promoted_candidates"] == 4
    assert value["window_counters"]["accepted_promoted_candidates"] == 4
    assert value["window_counters"]["committed_tokens"] == 5
    assert value["window_counters"]["discarded_early_tokens"] == 0
    assert value["window_unhidden_wait_ns"] == 20
    assert value["observation"]["bridge_mismatches"] == "NOT_OBSERVED"
    assert value["GPU_overlap"]["status"] == "UNKNOWN"
    assert eager_columns({"rolling_eager": value})["eager_promotions"] == 1


@pytest.mark.parametrize("mutation", ["counter", "pending", "owner", "committed"])
def test_invalid_eager_evidence_cannot_pass_summary(mutation):
    backend, runtime = eager_evidence()
    source = backend["rolling_eager"]
    if mutation == "counter":
        source["counters"]["promotions"] += 1
    elif mutation == "pending":
        source["pending_work"] = ["still-writing"]
    elif mutation == "owner":
        source["owner_stopped"] = False
    else:
        runtime["requests"] = [{"commits": [{"token_ids": [99]}]}]
    with pytest.raises(DataError):
        summarize_eager(backend, runtime)


def test_gpu_overlap_requires_native_cuda_intervals_and_ignores_host_overlap():
    host = {"host_start_ns": 100, "host_launch_end_ns": 200, "purpose": "eager"}
    devices = [{"device": {"forwards": [host]}}]
    assert eager_overlap(devices, {"forwards": [host]}, 0, 300)["status"] == "UNKNOWN"
    native = {**host, "start_lower_ns": 100, "start_upper_ns": 110,
              "end_lower_ns": 190, "end_upper_ns": 200, "gpu_event_ms": 0.00008}
    devices = [{"device": {"forwards": [native],
                            "identity": {"global_rank": rank, "gpu_uuid": f"GPU-{rank + 1}"}}}
               for rank in (0, 1)]
    draft = {"forwards": [native], "identity": {"gpu_uuid": "GPU-0"}}
    result = eager_overlap(devices, draft, 0, 300)
    assert result["status"] == "OBSERVED" and result["event_overlap_lower_ms"] > 0
    assert not result["host_overlap_is_gpu_evidence"]
    native["gpu_event_ms"] = 0
    with pytest.raises(DataError, match="native"):
        eager_overlap(devices, draft, 0, 300)


def test_existing_window_prefix_and_candidate_checks_execute_for_new_mode():
    runtime, backend, selected, opts = evidence("serial-eager")
    measured = timing(runtime, backend, selected, opts)
    assert measured["measurement_status"] == "PASS"
    assert measured["actual_target_batch"]["min"] == 16
    assert measured["decode_throughput_tok_s"] > 0
    broken = copy.deepcopy(runtime)
    broken["target_devices"][0]["rounds"][-1]["committed_token_ids"] = [999]
    with pytest.raises(DataError, match="accounting"):
        timing(broken, backend, selected, opts)


def test_eager_summary_status_and_bundle_retain_metrics_and_gpu_check(scan_root):
    selected = selected_point("serial-eager", 16)
    directory, saved = report(scan_root, selected)
    backend, runtime = eager_evidence()
    saved["rolling_eager"] = summarize_eager(backend, runtime)
    saved.update(eager_columns(saved))
    publish(directory / "light-summary.json", saved)
    (scan_root / "rolling-eager-gpu-check").mkdir()
    publish(scan_root / "rolling-eager-gpu-check/result.json", {"GPU_correctness": "PENDING"})
    publish(scan_root / "rolling-eager-gpu-check/state.json", {"state": "NOT_RUN"})
    result = decode_scan_cli.summary(scan_root)
    row = next(row for row in result["points"] if row["mode"] == "serial-eager")
    assert row["eager_started"] == 1 and row["eager_verified_candidates"] == 4
    assert row["rolling_eager"]["GPU_overlap"]["status"] == "UNKNOWN"
    with Path(result["artifact"]).with_suffix(".csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert next(row for row in rows if row["mode"] == "serial-eager")["eager_promotions"] == "1"
    status = fixed_cli.status(scan_root)
    assert status["results"][0]["eager_started"] == 1
    output = scan_root / "bundle.tar.gz"
    decode_scan_cli.bundle(scan_root, output)
    with tarfile.open(output) as archive:
        assert "rolling-eager-gpu-check/result.json" in archive.getnames()
        summary = json.load(archive.extractfile(str(Path(result["artifact"]).name)))
        assert any(row["mode"] == "serial-eager" for row in summary["points"])
    assert read_json(directory / "light-summary.json")["eager_accepted_candidates"] == 4
