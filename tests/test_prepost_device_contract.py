"""Actual startup/snapshot/timeline producers -> real scan qualifier -> archive replay.

CUDA model/events and prefilled workload are CPU fixtures. No identity producer,
report generator, existing preparation checks or qualification function is replaced.
"""

import copy
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_phase4_dual_uuid import harness as _hardware
from test_serving_decode_scan_results import evidence
from test_serving_s1 import request
from test_serving_s2 import profile
from test_serving_s2_results import fixture as native_fixture

from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
from specrhythm.phase4.vllm_remote import RemoteDraftProposer, _TargetRequest
from specrhythm.serving import decode_scan_results, fixed_logging, fixed_observe, s2_runtime
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.device_contract import PREPOST_MODES, qualify_prepost
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.ping_prepost_proposer import PingPrePostProposer
from specrhythm.serving.ping_prepost_window import PingPrePostWindow
from specrhythm.serving.prepost_proposer import PrePostProposer
from specrhythm.serving.s2_plan import sealed

hardware = _hardware


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def produced(tmp_path, monkeypatch, hardware):
    def build(mode, probe, *, root=None):
        base_path = root or tmp_path
        # Reuse existing validated process/backend evidence for the simulated model lifecycle.
        _, directory, _ = native_fixture(base_path / "base", monkeypatch, "serial")
        runtime, _, point, opts = evidence(
            "pingpong" if (mode.startswith("pingpong") or mode == "serial-k3") else "serial")
        opts.update(observation="original-live", identity_matching="linear")
        point.update(mode=mode, runtime_mode=mode, probe=probe)
        rows = [replace(request(i, 512), request_id=str(i)) for i in range(360)]
        path, manifest, _ = profile(base_path / "profile", rows=rows)
        manifest.pop("sha256")
        manifest.update(active_limit=16, fixed_diagnostic=dict(options=opts))
        manifest = sealed(manifest)
        write(path, manifest)
        runtime.update(
            point=point,
            probe=probe,
            target_requests_final=0,
            capacity={"resident_request_count": 360},
        )
        if probe:
            runtime.update(
                target_steps=[],
                stop_reason="capacity_probe",
                diagnostic_drain=dict(status="COMPLETE", settled_requests=360),
            )
            for row in runtime["requests"]:
                row.update(generated_token_ids=[10], commits=[])
        for row in runtime["requests"]:
            if row["admission_ns"] is not None:
                runtime["events"].append(
                    dict(
                        event="cancelled-resources-released",
                        request_id=row["request_id"],
                        timestamp_ns=runtime["end_ns"] - 1,
                    )
                )
            row["resources_released"] = True
        if (mode.startswith("pingpong-") or mode == "serial-k3") and not probe:
            w = PingPrePostWindow(opts, 16, True)
            for s in runtime["target_steps"]:
                w.ready(
                    s["start_ns"], population=runtime["decode_scan"]["window_initial_population"]
                )
                w.step_completed(s, s["end_ns"])
            w.end_ns = runtime["measurement_end_ns"]
            runtime["decode_scan"] = w.evidence()
        runtime["decode_scan"]["prefill_complete_ns"] = 0
        original_devices = copy.deepcopy(runtime["target_devices"])
        monkeypatch.setattr(fixed_observe, "install_host_observation", lambda: None)
        monkeypatch.setattr(fixed_logging, "_CURRENT", None)
        monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", "linear")
        monkeypatch.setenv("SR_S2_MODE", mode)
        monkeypatch.setenv("SR_S1_EXECUTION_MANIFEST", str(path))
        clock = NS(value=100)
        monkeypatch.setattr(fixed_observe, "time", NS(monotonic_ns=lambda: clock.value))

        class Event:
            def __init__(self, **kw):
                self.tick = 0

            def record(self):
                self.tick = clock.value

            def synchronize(self):
                pass

            def query(self):
                return True

            def elapsed_time(self, other):
                return (other.tick - self.tick) / 1e6

        for name in (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            setattr(hardware.cuda, name, lambda device=None: 1024)
        hardware.cuda.Event = Event
        hardware.cuda.mem_get_info = lambda: (8 * 1024**3, 80 * 1024**3)
        cls = (PingPrePostProposer if (mode.startswith("pingpong-") or mode == "serial-k3")
               else PrePostProposer)
        workers, startup = [], []
        for rank in (0, 1):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(rank + 1))
            worker = hardware.worker(rank=rank)
            p = object.__new__(
                cls
            )  # GPU model/transport construction substitute; real hook methods.
            p.tp_rank = rank
            p.torch = hardware.torch
            p.tp_group = NS(barrier=lambda: None)
            p.hooks_seen = dict(verify_start=0, verify_end=0)
            p.verify_phase_sequence = 0
            p.identity = FrozenPromptIdentityMap(
                {r.request_id: tuple(r.prompt_token_ids) for r in rows}
            )
            p.internal_to_stable = p.identity.internal_to_stable
            p.requests = {}
            worker.model_runner.drafter = p
            worker.model_runner.use_async_scheduling = False
            worker.model_runner._bookkeeping_sync = lambda *a: None
            worker.model_runner.input_batch = NS(req_ids=[])
            worker.model_runner.kv_cache_config = NS(num_blocks=100000, kv_cache_groups=[object()])
            worker.vllm_config.cache_config.block_size = 16
            worker.vllm_config.model_config = NS(
                enforce_eager=True,
                max_model_len=4096,
                dtype="bfloat16",
                get_vocab_size=lambda: 151936,
            )
            worker.vllm_config.scheduler_config.max_num_seqs = 512
            worker.vllm_config.scheduler_config.max_num_batched_tokens = 4096
            model = worker.get_model()
            model.register_forward_pre_hook = lambda f: None
            model.register_forward_hook = lambda f: None
            worker.get_model = lambda model=model: model
            startup.append(fixed_observe.target_startup(worker))
            workers.append(worker)
        reports = []
        final = []
        for rank, worker in enumerate(workers):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(rank + 1))
            monkeypatch.setattr(fixed_observe, "ROUNDS", [])
            monkeypatch.setattr(
                fixed_observe, "TARGET_ROWS", copy.deepcopy(original_devices[rank]["target_rows"])
            )
            p = worker.model_runner.drafter
            for s in runtime["target_steps"]:
                scheduled = {r["internal_request_id"]: [11, 12, 13, 14] for r in s["rows"]}
                for r in s["rows"]:
                    rid = r["request_id"]
                    prefix = tuple(rows[int(rid)].prompt_token_ids) + (10,) * (
                        r["context_length"] - 2
                    )
                    p.identity.bind(rid, prefix)
                    p.requests[rid] = _TargetRequest(
                        rid, tuple(rows[int(rid)].prompt_token_ids), 512, prefix, tuple(prefix[2:])
                    )
                    p.requests[rid].pending_proposal = Proposal(
                        PROTOCOL_VERSION,
                        rid,
                        0,
                        len(prefix),
                        token_prefix_hash(prefix),
                        (11, 12, 13, 14),
                        False,
                        1,
                        2,
                        1,
                        {},
                        {},
                    )
                # Underlying real Serial verifier increments the existing raw per-request counters.
                RemoteDraftProposer.on_target_verify_start(
                    p, request_ids=s["request_ids"], scheduled_spec_token_ids=scheduled
                )
                worker.model_runner.input_batch.req_ids = s["request_ids"]
                clock.value = s["start_ns"] + 2
                worker.fixed_timeline.before(None, None)
                clock.value = s["end_ns"] - 2
                worker.fixed_timeline.after(None, None, None)
                RemoteDraftProposer.on_target_verify_end(
                    p,
                    request_ids=s["request_ids"],
                    scheduled_spec_token_ids=scheduled,
                    sampled_token_ids=[],
                )
            for r in original_devices[rank]["rounds"] if not probe else []:
                fixed_observe.capture({"schema_version": "specrhythm.phase4-round-event.v1", **r})
            final.append(s2_runtime.target_snapshot(worker))
            reports.append(fixed_observe.target_report(worker))
        runtime.update(target_devices=reports, target_final_memory=final)
        actual = dict(
            target_worker_ranks=startup,
            ranks=[r["s2_capacity"] for r in startup]
            + [
                dict(
                    role="draft",
                    mode=mode,
                    physical_gpu_id=0,
                    gpu_uuid="GPU-DRAFT",
                    block_size=16,
                    num_gpu_blocks=100000,
                )
            ],
            checks=[dict(valid=True, pool_size=360, active_limit=16)] * 3,
            execution_sha256=manifest["sha256"],
            workload_sha256=manifest["workload_sha256"],
            point=point,
        )
        backend = read_json(directory / "draft-backend-report.json")
        backend["fixed_device"] = dict(
            identity=dict(role="draft", physical_gpu_id=0, gpu_uuid="GPU-DRAFT"), forwards=[]
        )
        life = read_json(directory / "process-lifecycle.json")
        life["owned_cleanup_completed"] = True
        write(directory / "process-lifecycle.json", life)
        write(directory / "actual-capacity.json", actual)
        write(directory / "runtime.json", runtime)
        write(directory / "draft-backend-report.json", backend)
        write(
            directory / "resident-pool.json",
            dict(
                selected_requests=360,
                resident_requests=360,
                bootstrap_terminal_requests=0,
                prefill_setup_ns=1,
                target=dict(
                    initial={str(i): {} for i in range(360)}, timed_reprefill_allowed=False
                ),
                draft=dict(rows={str(i): {} for i in range(360)}),
            ),
        )
        return NS(
            path=path,
            directory=directory,
            runtime=runtime,
            backend=backend,
            actual=actual,
            point=point,
            probe=probe,
            workers=workers,
            hardware=hardware,
            native_clock=clock,
        )

    return build


@pytest.mark.parametrize("mode", PREPOST_MODES)
@pytest.mark.parametrize("probe", [True, False])
def test_production_snapshots_and_report_qualifier(mode, probe, produced):
    h = produced(mode, probe)
    queries = list(h.hardware.calls)
    result = decode_scan_results.summarize(h.path, h.directory, h.point, probe=probe)
    assert result["valid"], (result.get("errors"), result.get("primary_error"))
    assert result["device_identity_qualification"]["verification_requests"] == (
        0 if probe else sum(s["B"] for s in h.runtime["target_steps"])
    )
    assert all("dual_uuid_query" not in r for r in h.runtime["target_final_memory"])
    assert not probe or result["measurement_status"] == "NOT_APPLICABLE"
    assert h.hardware.calls == queries  # qualification is offline, never a UUID subprocess
    assert len(queries) == 4  # existing startup/final snapshot per TP rank only


def test_original_summarize_reproduces_first_exception_from_real_producers(produced):
    h = produced("pingpong-prepost3", True)
    scope = dict(vars(decode_scan_results))
    source = Path("tests/fixtures/device_contract/bc908be-summarize.py.txt")
    exec(compile(source.read_text(), str(source), "exec"), scope)
    caught = []

    def trace(frame, event, arg):
        if event == "exception" and frame.f_code.co_filename == str(source):
            caught.append(
                (frame.f_code.co_name, frame.f_lineno, type(arg[1]).__name__, str(arg[1]))
            )
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = scope["summarize"](h.path, h.directory, h.point, probe=True)
    finally:
        sys.settrace(previous)
    assert result["errors"] == ["'dual_uuid_query'"] and not result["valid"]
    # Python 3.12 reports this in the containing frame. Assert the exact original
    # source location and exception, rather than an interpreter-specific frame name.
    assert source.read_text().splitlines()[caught[0][1] - 1].strip() == (
        'ranks = [x["dual_uuid_query"] for x in r["target_final_memory"]]')
    assert caught[0][2:] == ("KeyError", "'dual_uuid_query'")
    assert decode_scan_results.summarize(h.path, h.directory, h.point, probe=True)["valid"]


@pytest.mark.parametrize("mode", PREPOST_MODES[2:])
def test_joint_correctness_report_uses_same_strict_contract(mode, produced):
    from specrhythm.serving.fixed_results import summarize

    h = produced(mode, False)
    h.point["prepost_correctness"] = True
    h.point.pop("scan", None)
    h.runtime.pop("decode_scan", None)
    h.runtime["point"] = h.point
    del h.runtime["target_final_memory"][1]["gpu_uuid"]
    write(h.directory / "runtime.json", h.runtime)
    result = summarize(h.path, h.directory, h.point)
    assert not result["valid"] and result["qualification_status"] == "FAILED"
    details = result["primary_error"]
    assert details["stage"] == "correctness" and details["rank"] == 1
    assert details["field"] == "gpu_uuid"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_final",
        "empty_uuid",
        "missing_rank",
        "duplicate_rank",
        "wrong_physical",
        "native_uuid",
        "draft_alias",
        "declared_mode",
        "actual_path",
        "missing_contract",
        "missing_forward",
        "wrong_request",
        "wrong_proposal",
        "hook_count",
        "capacity_mode",
        "zero_formal",
    ],
)
def test_missing_or_conflicting_real_evidence_fails_with_context(produced, fault):
    h = produced("pingpong-eager-prepost3", False)
    r, b, a = h.runtime, h.backend, h.actual
    if fault == "missing_final":
        del r["target_final_memory"]
    elif fault == "empty_uuid":
        r["target_final_memory"][0]["gpu_uuid"] = ""
    elif fault == "missing_rank":
        r["target_devices"].pop()
    elif fault == "duplicate_rank":
        r["target_final_memory"][1]["global_rank"] = 0
    elif fault == "wrong_physical":
        r["target_final_memory"][1]["physical_gpu_id"] = 0
    elif fault == "native_uuid":
        r["target_devices"][0]["device"]["identity"]["gpu_uuid"] = "wrong"
    elif fault == "draft_alias":
        b["fixed_device"]["identity"]["gpu_uuid"] = r["target_final_memory"][0]["gpu_uuid"]
    elif fault == "declared_mode":
        r["target_final_memory"][0]["device_evidence_contract"]["mode"] = "pingpong"
    elif fault == "actual_path":
        r["target_final_memory"][0]["device_evidence_contract"]["proposer_path"] = (
            "DualBatchRemoteProposer"
        )
    elif fault == "missing_contract":
        del r["target_final_memory"][0]["device_evidence_contract"]
    elif fault == "missing_forward":
        r["target_devices"][1]["device"]["forwards"].pop()
    elif fault == "wrong_request":
        r["target_devices"][1]["device"]["forwards"][0]["internal_request_ids"] = ["wrong"]
    elif fault == "wrong_proposal":
        r["target_devices"][0]["rounds"][0]["proposal_token_ids"] = [801] * 4
    elif fault == "hook_count":
        r["target_final_memory"][0]["device_evidence_contract"]["verification_hooks"][
            "verify_end"
        ] -= 1
    elif fault == "capacity_mode":
        a["ranks"][0]["mode"] = "pingpong"
    else:
        r["target_steps"] = []
    write(h.directory / "runtime.json", r)
    write(h.directory / "draft-backend-report.json", b)
    write(h.directory / "actual-capacity.json", a)
    result = decode_scan_results.summarize(h.path, h.directory, h.point, probe=False)
    assert not result["valid"] and result["qualification_status"] == "FAILED"
    assert result["effective_exit_code"] == 0  # report rejection is not a process failure
    details = result["primary_error"]
    assert all(k in details for k in ("mode", "stage", "artifact", "rank", "field", "expected"))
    assert details["failure_layer"] == "report_qualification"
    assert details["mode"] == h.point["mode"] and details["stage"] == "performance"


@pytest.mark.parametrize("mode", PREPOST_MODES[2:])
def test_produce_export_reload_same_identity_contract(mode, produced, tmp_path):
    h = produced(mode, True)
    baseline = decode_scan_results.summarize(h.path, h.directory, h.point, probe=True)
    assert baseline["valid"], baseline.get("errors")
    write(h.directory / "light-summary.json", baseline)
    archive = tmp_path / "delivery.tar.gz"
    status = export(h.directory, archive, first_code=23, stage="later_failure")
    assert status["export_status"] == "COMPLETE" and status["first_exit_code"] == 23
    with tarfile.open(archive) as pack:
        inv = json.load(pack.extractfile("inventory.json"))

        def read(name):
            return json.load(pack.extractfile(inv["logical_paths"][name]))

        r, b, a = (
            read(n) for n in ("runtime.json", "draft-backend-report.json", "actual-capacity.json")
        )
    assert r["target_final_memory"] == h.runtime["target_final_memory"]
    assert qualify_prepost(r, b, a, mode, probe=True) == baseline["device_identity_qualification"]
    del h.runtime["target_final_memory"]
    write(h.directory / "runtime.json", h.runtime)
    missing = export(h.directory, tmp_path / "missing.tar.gz", first_code=23)
    assert missing["export_status"] == "INCOMPLETE"
    row = next(r for r in missing["inventory"] if r["path"] == "runtime.json")
    assert row["required_fields_missing"] == ["target_final_memory"]
    with pytest.raises(DataError, match="target_final_memory"):
        qualify_prepost(h.runtime, b, a, mode, probe=True)


def test_actual_qualification_failure_keeps_process_exit_and_single_package(produced, tmp_path):
    from specrhythm.serving.execution_failure import summarize
    from specrhythm.serving.fixed_artifacts import retained_report

    h = produced("pingpong-prepost3", True)
    del h.runtime["target_final_memory"][1]["device_evidence_contract"]
    write(h.directory / "runtime.json", h.runtime)
    result = decode_scan_results.summarize(h.path, h.directory, h.point, probe=True)
    result = retained_report(h.directory, result)
    root = tmp_path / "delivery/points/pingpong-prepost3"
    write(root / "runs/capacity/light-summary.json", result)
    failure = summarize(root, 1, "capacity")
    assert failure["first_exit_code"] == 1
    assert failure["failure_layer"] == "report_qualification"
    run = failure["run_qualifications"][0]
    assert run["effective_exit_code"] == 0 and run["cleanup_status"] == "PASS"
    assert run["qualification_status"] == "FAILED"
    assert run["primary_error"]["rank"] == 1
    assert run["primary_error"]["field"] == "device_evidence_contract"
    delivery = root.parent.parent
    write(delivery / "first-failure.json", failure)
    archive = tmp_path / "pingpong-prepost3-delivery-failure.tar.gz"
    status = export(delivery, archive, first_code=1, stage="capacity")
    assert status["first_exit_code"] == 1
    with tarfile.open(archive) as pack:
        paths = json.load(pack.extractfile("inventory.json"))["logical_paths"]
        assert json.load(pack.extractfile(paths["first-failure.json"])) == failure
