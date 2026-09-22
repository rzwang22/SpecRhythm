"""Production owner/controller, full live audits, scoped publication and report contracts."""

import copy
import json
import threading
from types import SimpleNamespace as NS

import pytest
from test_dual_batch import fixed_schedulers as _fixed
from test_dual_batch import s2_schedulers as _s2
from test_dual_batch import target_pool as _pool
from test_k3 import machine
from test_k3_pipeline import close
from test_k3_resident_schedule import prepare
from test_prepost_protocol import feedback
from test_serial_eager_owner import verify_row

from specrhythm.serving import s2_pool
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.ordinary_cpu_report import aggregate, compare_smoke, window_metrics
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.shared_control import run_target, validate_mode

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


@pytest.mark.parametrize("mode", ["serial-k3", "pingpong-k3"])
def test_shared_owner_controller_rejection_success_and_natural_close(monkeypatch, mode):
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "shared-command")
    monkeypatch.setenv("SR_S2_MODE", mode)
    ready = {rid: threading.Event() for rid in ("a", "b")}

    def build():
        m = machine(False, mode=mode)
        ping = m._ping

        def observed(event, **values):
            ping(event, **values)
            if event == "ready":
                ready[values["request_id"]].set()

        m._ping = observed
        return m

    owner = K3Owner(build, timeout_seconds=3)
    ctl = PingPrePostController(mode)
    calls = []

    def rpc(op, payload):
        calls.append(op)
        return json.loads(json.dumps(owner.call(op, json.loads(json.dumps(payload)))))

    client = NS(call=rpc)
    clock = NS(rows={r: {"state": "ACTIVE"} for r in ("a", "b")})
    try:
        for reject in [False, True, True, False]:
            # Synchronize through the existing owner command, never a fake READY.
            assert client.call("k3_idle", {})["idle"]
            value = ctl.select(clock, client)
            assert len(value["claims"]) == (2 if mode == "serial-k3" else 1)
            for claim in value["claims"]:
                p = claim["proposal"]
                assert len(p["proposal_token_ids"]) == 3
                client.call("eager_enqueue", {"requests": [verify_row(p)]})
                prefix = tuple(owner.machine.requests[p["request_id"]].committed_token_ids)
                row, _ = feedback(p, prefix, reject=reject)
                ready[p["request_id"]].clear()
                client.call("pp_feedback", {"synchronizations": [row]})
                assert ready[p["request_id"]].wait(3)
                before = tuple(owner.machine.requests[p["request_id"]].committed_token_ids)
                client.call("pp_feedback", {"synchronizations": [row]})
                assert tuple(owner.machine.requests[p["request_id"]].committed_token_ids) == before
        assert "pp_admit_command" in calls and not owner.machine.works
    finally:
        close(owner)
    assert owner.closed


def test_serial_shared_gate_cannot_be_bypassed_and_old_guard_stays(monkeypatch):
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "dual-batch")
    with pytest.raises(ValueError, match="ordinary pingpong"):
        validate_mode("serial-k3")
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "shared-command")
    validate_mode("serial-k3")
    for mode in ("serial-eager-k3", "pingpong-eager-k3", "serial"):
        with pytest.raises(ValueError, match="ordinary"):
            validate_mode(mode)
    m = machine(False, mode="serial-k3", complete=False)
    owner = object.__new__(K3Owner)
    owner.machine = m
    try:
        with pytest.raises(ValueError, match="idle"):
            owner._dispatch(
                "pp_admit_command",
                dict(opportunity=0, capacity=16, normal_cohort="A", active_request_ids=["a", "b"]),
            )
    finally:
        from test_prepost_protocol import cleanup

        cleanup(m)


@pytest.mark.parametrize("mode", ["serial-k3", "pingpong-k3"])
def test_real_scheduler_full_audits_use_block_sets_and_keep_live_failures(
    target_pool, monkeypatch, mode
):
    scheduler, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", "shared-command")
    monkeypatch.setenv("SR_K3_TARGET_CPU", "block-sets")
    scheduler.s2_pool = s2_pool.ResidentPoolAudit("target")
    prepare(scheduler, packet, path)
    count = scheduler.s2_pool.checks
    output = run_target(NS(step=scheduler.schedule))
    assert output.num_scheduled_tokens
    assert scheduler.s2_pool.checks == count + 2
    assert scheduler.s2_pool.report()["ownership_check"] == "block-sets"
    # Actual live allocator corruption, not a fabricated qualifier result.
    scheduler.kv_cache_manager = NS(get_block_ids=lambda _: [[1]])
    with pytest.raises(ValueError, match="shared"):
        scheduler.s2_pool.check(scheduler.physical_rows(), packet["requests"])


@pytest.mark.parametrize(
    "fault",
    [None, "duplicate", "shared", "bool", "float", "negative", "frontier", "evict", "replace"],
)
def test_both_full_audit_implementations_detect_same_faults(monkeypatch, fault):
    initial = {
        str(i): dict(materialized_tokens=16, block_ids=[[i * 2, i * 2 + 1]], prefix_sha256=str(i))
        for i in range(360)
    }
    states = {i: {"state": "ACTIVE"} for i in initial}
    rows = copy.deepcopy(initial)
    if fault in ("duplicate", "shared", "bool", "float", "negative", "replace"):
        rows["0"]["block_ids"] = [
            dict(
                duplicate=[0, 0],
                shared=[2, 3],
                bool=[False, 1],
                float=[0.0, 1],
                negative=[-1, 1],
                replace=[800, 801],
            )[fault]
        ]
    elif fault == "frontier":
        rows["0"]["materialized_tokens"] = 0
    elif fault == "evict":
        del rows["0"]
    results = []
    for policy in ("reference", "block-sets"):
        monkeypatch.setenv("SR_K3_TARGET_CPU", policy)
        audit = s2_pool.ResidentPoolAudit("target")
        audit.check(initial, states, freeze=True)
        try:
            audit.check(rows, states)
        except ValueError as error:
            results.append(str(error))
        else:
            report = audit.report()
            report.pop("ownership_check")
            results.append(report)
    assert results[0] == results[1]
    assert isinstance(results[0], dict) is (fault is None)


def test_block_loop_structural_reduction_without_skipping_checks(monkeypatch):
    rows = {"a": dict(materialized_tokens=2048, block_ids=[list(range(128))])}
    old = s2_pool.require
    calls = []
    monkeypatch.setattr(s2_pool, "require", lambda *a, **kw: (calls.append(a), old(*a, **kw))[1])
    for policy, count in [("reference", 130), ("block-sets", 3)]:
        monkeypatch.setenv("SR_K3_TARGET_CPU", policy)
        calls.clear()
        s2_pool.ResidentPoolAudit("target").check(rows, {})
        assert len(calls) == count
    assert s2_pool.ResidentPoolAudit("draft").ownership_check == "reference"


def test_smoke_first_difference_and_window_tpot_no_fake_completions():
    rows = [
        dict(
            case=k,
            target_dispatch=k,
            outputs={"r": dict(tokens=[1, 2], finish_reason="length")},
            run_directory="/" + k,
        )
        for k in ["P0", "P1", "S0", "S1"]
    ]
    assert compare_smoke(rows, cpu_comparison=True)["status"] == "PASS"
    rows[1]["outputs"]["r"]["tokens"] = [1, 3]
    failed = compare_smoke(rows, cpu_comparison=True)
    assert failed["status"] == "FAILED"
    assert failed["differences"][0]["actual"] == "P1"
    assert failed["differences"][0]["first_difference_index"] == 1
    r = dict(
        request_id="r",
        state="FINISHED",
        completion_ns=30_000_000,
        generated_token_ids=[1, 2, 3, 4],
        commits=[
            dict(timestamp_ns=10_000_000, token_ids=[2]),
            dict(timestamp_ns=30_000_000, token_ids=[3, 4]),
        ],
    )
    value = window_metrics(
        dict(
            measurement_start_ns=0,
            measurement_end_ns=40_000_000,
            requests=[r, dict(state="CANCELLED")],
        )
    )
    assert value["natural_completions_in_window"] == 1
    assert value["full_request_tpot_ms"]["mean"] == 10
    windows = [
        dict(
            case="P0",
            target_dispatch="dual-batch",
            request_measurement=value,
            points=[dict(committed_tokens=n, window_ms=t, throughput_tok_s=n / (t / 1000))],
        )
        for n, t in [(100, 1000), (100, 2000)]
    ]
    assert aggregate(windows)["P0"]["pooled_throughput_tok_s"] == 200 / 3


def test_recovery_classification_requires_calibrated_whole_call_before_other_gpu(monkeypatch):
    from specrhythm.serving import k3_dispatch_evidence
    from specrhythm.serving.ordinary_cpu_report import recovery_during_prepare

    groups = [
        dict(
            index=1,
            step={"window": True},
            claims={"b": dict(claimed_ns=10, home_cohort="B")},
            native=[dict(start_lower_ns=50), dict(start_lower_ns=51)],
        )
    ]
    monkeypatch.setattr(k3_dispatch_evidence, "target_groups", lambda r, e: groups)
    native = [dict(purpose="normal", B=1, host_start_ns=21, start_lower_ns=22, end_upper_ns=40)]
    backend = dict(
        prepost={"pingpong": {"home_cohorts": {"a": "A"}}},
        fixed_device={"forwards": native},
        prepost_physical={
            "forwards": [
                dict(
                    start_ns=20,
                    end_ns=45,
                    B=1,
                    purpose="normal",
                    physical_forward_id="p",
                    bindings=[dict(request_id="a", source="rejection")],
                )
            ]
        },
    )
    runtime = dict(measurement_start_ns=0, measurement_end_ns=100)
    result = recovery_during_prepare(runtime, backend)
    assert result["finished_wholly_in_other_home_claim_to_GPU"] == 1
    native[0]["end_upper_ns"] = 52  # Uncertainty crosses GPU start: cannot claim before.
    result = recovery_during_prepare(runtime, backend)
    assert result["finished_wholly_in_other_home_claim_to_GPU"] == 0
    assert result["unclassified_calls"] == 1
    backend["fixed_device"]["forwards"] = []
    assert recovery_during_prepare(runtime, backend)["status"] == "INCOMPLETE"
