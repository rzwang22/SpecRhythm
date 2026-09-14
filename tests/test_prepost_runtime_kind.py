"""Real drive/clock/drain and startup/report producers; GPU/IPC outputs are CPU fixtures."""

import copy
import json
import shutil
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_prepost_device_contract import hardware as _hardware
from test_prepost_device_contract import produced as _produced
from test_prepost_device_contract import write
from test_serving_s1 import request
from test_serving_s2 import profile

from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
from specrhythm.phase4.vllm_remote import RemoteDraftProposer, _TargetRequest
from specrhythm.serving import fixed_observe, fixed_runtime
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.device_contract import qualify_prepost
from specrhythm.serving.fixed_plan import capacity_metadata
from specrhythm.serving.fixed_results import summarize
from specrhythm.serving.ping_prepost import MODES, SCHEDULED_MODES
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import sealed

hardware, produced = _hardware, _produced


@pytest.fixture
def driven(produced, monkeypatch, tmp_path):
    calls = []

    def build(mode, stage, driver=None, *, full_run=False):
        base = tmp_path / str(len(calls))
        calls.append((mode, stage))
        h = produced(mode, True, root=base)  # Actual startup/snapshot, no decode hooks yet.
        directory = base / "drive-output"
        shutil.copytree(
            h.directory,
            directory,
            ignore=shutil.ignore_patterns("runtime.json", "arrival-output-events.json"),
        )
        h.directory = directory
        correct, probe = stage == "correctness", stage == "capacity_probe"
        scan = not correct
        rows = [
            replace(request(i, 2 if correct else 512), request_id=str(i))
            for i in range(16 if correct else 360)
        ]
        path, m, _ = profile(base / "drive-profile", rows=rows)
        m.pop("sha256")
        m.update(active_limit=16, fixed_diagnostic=read_json(h.path)["fixed_diagnostic"])
        m["fixed_diagnostic"].update(cohorts={"A": [], "B": []})
        m["fixed_diagnostic"]["capacity"] = {
            mode: capacity_metadata(
                mode, active_limit=16, resident_requirement=len(rows), target_sequence_limit=512
            )
        }
        opts = m["fixed_diagnostic"]["options"]
        opts.update(
            setup_timeout=900,
            drain_timeout=60,
            samples=4096 if correct else None,
            warmup_steps=0 if correct else 2,
            window_seconds=900 if correct else 30,
        )
        m["trace"].pop("sha256")
        for row in m["trace"]["rows"]:
            row["arrival_offset_seconds"] = 0
        m["trace"] = sealed(m["trace"])
        m = sealed(m)
        write(path, m)
        _, definitions = load_s2(str(path))
        by_id = {r.request_id: r for r in definitions}
        point = dict(
            h.point,
            probe=probe,
            scan=scan,
            prepost_correctness=correct,
            kind="joint-correctness" if correct else "decode-scan",
        )
        scheduler = NS(s2_steps=[], requests=dict.fromkeys(by_id), s2_pool=NS(report=lambda: {}))
        warm = {
            rid: NS(
                prefix_version=1,
                logical_committed_prefix_token_ids=d.prompt_token_ids + (10,),
                logical_committed_prefix_sha256=token_prefix_hash(d.prompt_token_ids + (10,)),
            )
            for rid, d in by_id.items()
        }
        generated = {rid: [10] for rid in by_id}
        released, native_rows, rounds = {}, [], []
        ticks = [time.monotonic_ns()]

        def now():
            ticks[0] += 1000
            return ticks[0]

        monkeypatch.setattr(time, "monotonic_ns", now)
        monkeypatch.setattr(time, "monotonic", lambda: now() / 1e9)
        monkeypatch.setenv("SR_S2_CONTROL", str(h.directory / "s2-control.json"))
        monkeypatch.setenv("SR_S2_RUN_DIRECTORY", str(h.directory))
        monkeypatch.delenv("SR_FIXED_SCAN_SETUP_DEADLINE_NS", raising=False)

        class Client:
            def call(self, operation, payload):
                if operation == "k3_idle":
                    return dict(idle=True)
                if operation == "status":
                    return dict(inflight_request_ids=[], failures={})
                if operation == "pp_admit":
                    return dict(
                        claims=[dict(request_id=r) for r in payload["active_request_ids"][:8]]
                    )
                if operation in ("pp_register", "pp_stop"):
                    return {}
                if operation == "finish_request":
                    released[payload["request_id"]] = now()
                    return {}
                if operation == "diagnostic_settle":
                    rid = payload["request_id"]
                    released.setdefault(rid, now())
                    terminal = payload["natural_terminal"]
                    return dict(
                        request_id=rid,
                        released=True,
                        new_proposals_generated=0,
                        authoritative_prefix_hash=payload["committed_prefix_hash"],
                        authoritative_prefix_count=len(payload["committed_prefix"]),
                        natural_terminal=terminal,
                        disposition="NATURAL_TERMINAL" if terminal else "DIAGNOSTIC_CANCELLED",
                        resources_released_ns=released[rid],
                    )
                assert operation == "shutdown"
                return dict(shutdown=True)

        class Engine:
            def step(self):
                packet = read_json(h.directory / "s2-control.json")
                ids = [r["request_id"] for r in packet["pp_admission"]["claims"]]
                layout, output = [], []
                for rid in ids:
                    prefix = by_id[rid].prompt_token_ids + tuple(generated[rid])
                    n = len(prefix)
                    layout.append(
                        dict(
                            request_id=rid,
                            internal_request_id=rid,
                            context_length=n,
                            candidate_positions=1,
                            query_positions=2,
                            base_root_positions=1,
                            position_start=n - 1,
                            position_end_exclusive=n + 1,
                        )
                    )
                    native_rows.append(
                        dict(
                            request_id=rid,
                            context_length=n,
                            proposal_token_ids=[11],
                            query_length=2,
                            position_ids=[n - 1, n],
                            target_forward_start_ns=now(),
                            structural_errors=[],
                        )
                    )
                    for worker in h.workers:
                        p = worker.model_runner.drafter
                        p.identity.bind(rid, prefix)
                        p.requests[rid] = _TargetRequest(
                            rid,
                            by_id[rid].prompt_token_ids,
                            by_id[rid].maximum_new_tokens,
                            prefix,
                            tuple(generated[rid]),
                        )
                        p.requests[rid].pending_proposal = Proposal(
                            PROTOCOL_VERSION,
                            rid,
                            len(generated[rid]) - 1,
                            n,
                            token_prefix_hash(prefix),
                            (11,),
                            False,
                            now(),
                            now(),
                            1,
                            {},
                            {},
                        )
                    rounds.append(
                        dict(
                            request_id=rid,
                            parent_prefix_len=n,
                            proposal_token_ids=[11],
                            accepted_draft_token_ids=[11],
                            rejected_draft_token_ids=[],
                            target_correction_token_ids=[],
                            target_bonus_token_ids=[],
                            committed_token_ids=[11],
                            accepted_draft_tokens=1,
                            rejected_draft_tokens=0,
                        )
                    )
                    generated[rid].append(11)
                    output.append(
                        NS(
                            request_id=rid,
                            finished=correct,
                            outputs=[
                                NS(
                                    token_ids=list(generated[rid]),
                                    finish_reason="length" if correct else None,
                                )
                            ],
                        )
                    )
                    if correct:
                        del scheduler.requests[rid]
                for worker in h.workers:
                    p = worker.model_runner.drafter
                    RemoteDraftProposer.on_target_verify_start(
                        p, request_ids=ids, scheduled_spec_token_ids={r: [11] for r in ids}
                    )
                    worker.model_runner.input_batch.req_ids = ids
                    h.native_clock.value = now()
                    worker.fixed_timeline.before(None, None)
                    h.native_clock.value = now() + 100
                    worker.fixed_timeline.after(None, None, None)
                    RemoteDraftProposer.on_target_verify_end(
                        p,
                        request_ids=ids,
                        scheduled_spec_token_ids={r: [11] for r in ids},
                        sampled_token_ids=[],
                    )
                ticks[0] += 6_000_000_000
                scheduler.s2_steps.append(
                    dict(request_ids=ids, B=len(ids), rows=layout, cohort="A")
                )
                return output

            def abort_request(self, ids):
                for rid in ids:
                    scheduler.requests.pop(rid)

            def has_unfinished_requests(self):
                return bool(scheduler.requests)

        def rpc(callback, **kw):
            result = []
            for rank, worker in enumerate(h.workers):
                monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(rank + 1))
                monkeypatch.setattr(fixed_observe, "ROUNDS", rounds if rank == 0 else [])
                monkeypatch.setattr(fixed_observe, "TARGET_ROWS", native_rows if rank == 0 else [])
                result.append(callback(worker))
            return result

        llm = NS(collective_rpc=rpc)
        monkeypatch.setattr(
            fixed_runtime,
            "prepare_resident",
            lambda *a, **kw: (
                Engine(),
                scheduler,
                Client(),
                warm,
                {rid: dict(token=10, terminal=False) for rid in by_id},
                None,
            ),
        )
        if mode.endswith("-k3"):
            from specrhythm.serving.k3_capacity import budgets, check

            h.actual["metadata"] = m["fixed_diagnostic"]["capacity"][mode]
            h.actual["capacity_request_budgets"] = budgets(definitions)
            h.actual["checks"] = [check(definitions, r, mode=mode, active_limit=16,
                                      metadata=h.actual["metadata"]) for r in h.actual["ranks"]]
            h.actual.update(execution_sha256=m["sha256"], point=point,
                            workload_sha256=m["workload_sha256"])
            write(h.directory / "actual-capacity.json", h.actual)
        if full_run:
            # Substitute hardware creation, then execute run -> capacity_for -> real
            # drive -> immutable report. No mocked capacity function or PASS result.
            from specrhythm.phase4 import stock_vllm

            monkeypatch.setattr(stock_vllm, "worker_batch_invariant_evidence",
                                lambda worker: dict(batch_invariant_effective=True))
            cfg = h.workers[0].vllm_config
            cfg.scheduler_config.max_num_seqs = 512
            cfg.scheduler_config.max_num_batched_tokens = 4096
            cfg.speculative_config = NS(num_speculative_tokens=3)
            llm.llm_engine = NS(vllm_config=cfg, engine_core=NS(shutdown=lambda: None))
            config = NS(logprobs=1, target=NS(tensor_parallel_size=2, physical_gpu_ids=(1, 2)))
            binding_errors = fixed_runtime.validate_worker_ranks(h.actual["target_worker_ranks"],
                                                                 config.target)
            assert not binding_errors, binding_errors
            monkeypatch.setattr(fixed_runtime, "configure", lambda *a: (config, m, definitions))
            monkeypatch.setattr(fixed_runtime, "make_engine", lambda *a, **kw: llm)
            write(h.directory / "draft-startup.json", dict(s2_capacity=h.actual["ranks"][-1],
                fixed_engine_limits=dict(max_num_seqs=128, max_num_batched_tokens=4096)))
            # Discard the fixture; the real producer must write this report.
            (h.directory / "actual-capacity.json").unlink()
            fixed_runtime.run(base, path, h.directory, point, probe=probe)
            h.actual = read_json(h.directory / "actual-capacity.json")
        else:
            report = (driver or fixed_runtime.drive)(
                llm, m, definitions, h.directory, point, opts, probe=probe
            )
            # The same immutable serializer used by fixed_runtime.run, no injected probe field.
            write_once(h.directory / "runtime.json", report)
        return NS(
            **{
                **vars(h),
                "path": path,
                "runtime": read_json(h.directory / "runtime.json"),
                "point": point,
                "stage": stage,
                "probe": probe,
            }
        )

    return build


@pytest.mark.parametrize("mode", SCHEDULED_MODES)
@pytest.mark.parametrize("stage", ["capacity_probe", "correctness", "performance"])
def test_actual_drive_serialization_contract_and_archive(mode, stage, driven, tmp_path):
    h = driven(mode, stage)
    assert h.runtime["probe"] is h.point["probe"] is (stage == "capacity_probe")
    assert ("decode_scan" in h.runtime) == (stage != "correctness")
    contract = qualify_prepost(h.runtime, h.backend, h.actual, mode, probe=h.probe, stage=stage)
    assert contract["status"] == "PASS"
    if stage == "correctness":
        result = summarize(h.path, h.directory, h.point)
        assert result["valid"], result.get("primary_error")
    archive = tmp_path / "runtime-delivery.tar.gz"
    export(h.directory, archive, first_code=1)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        r, b, a = [
            json.load(t.extractfile(paths[n]))
            for n in ("runtime.json", "draft-backend-report.json", "actual-capacity.json")
        ]
    assert r["probe"] is h.probe
    assert qualify_prepost(r, b, a, mode, probe=h.probe, stage=stage) == contract


@pytest.mark.parametrize(
    "fault", ["missing", "string", "integer", "point", "stage", "scan", "native"]
)
def test_non_scan_report_contract_negative(fault, driven):
    h = driven(MODES[0], "correctness")
    r = copy.deepcopy(h.runtime)
    if fault == "missing":
        del r["probe"]
    elif fault == "string":
        r["probe"] = "false"
    elif fault == "integer":
        r["probe"] = 0
    elif fault == "point":
        r["point"]["probe"] = True
    elif fault == "stage":
        r["point"]["prepost_correctness"] = False
    elif fault == "scan":
        r["point"]["scan"] = True
    else:
        r["target_devices"][1]["device"]["forwards"].clear()
    with pytest.raises(DataError):
        qualify_prepost(r, h.backend, h.actual, MODES[0], stage="correctness")


def test_original_non_scan_drive_serializes_missing_probe(driven):
    def old_driver(*args, **kwargs):
        scope = dict(vars(fixed_runtime))  # Includes the current CPU hardware replacements.
        source = Path("tests/fixtures/runtime_kind/429956-drive.py.txt")
        exec(compile(source.read_text(), str(source), "exec"), scope)
        return scope["drive"](*args, **kwargs)

    h = driven(MODES[0], "correctness", driver=old_driver)
    assert h.runtime["point"]["probe"] is False and "probe" not in h.runtime
    result = summarize(h.path, h.directory, h.point)
    assert result["valid"] is False
    assert result["primary_error"]["field"] == "probe"
    assert result["failure_layer"] == "report_qualification"


@pytest.mark.parametrize(
    "point,probe",
    [
        ({}, 0),
        ({}, "false"),
        ({"probe": 0}, False),
        ({"probe": True}, False),
        ({"scan": 1}, False),
        ({"prepost_correctness": "true"}, False),
        ({"prepost_correctness": True}, True),
        ({"prepost_correctness": True, "scan": True}, False),
    ],
)
def test_invocation_conflicts_fail_before_model_preparation(point, probe):
    with pytest.raises(DataError):
        fixed_runtime.drive(None, None, None, None, point, None, probe=probe)
