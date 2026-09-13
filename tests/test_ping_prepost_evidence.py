"""Collect actual owner/backend records, then exercise strict report joins and single export."""

import copy
import json
import tarfile
import time

import pytest
from test_ping_prepost import admit, cleanup, machine, run_work
from test_prepost_protocol import feedback
from test_serial_eager_owner import verify_row

from specrhythm.continuation.prepost_records import Records
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.ping_prepost_evidence import mechanism


def collected(eager=True):
    m = machine(eager=eager)
    runtime = dict(
        point={"mode": "pingpong-eager-prepost3" if eager else "pingpong-prepost3"},
        target_steps=[],
        measurement_start_ns=time.monotonic_ns(),
    )
    samples = Records()
    target_native = []
    for cycle in range(5):
        start = time.monotonic_ns()
        batch = admit(m, "A" if cycle % 2 == 0 else "B")
        claims = batch["claims"]
        m.verify_start([verify_row(c["proposal"]) for c in claims])
        run_work(m)
        gpu_end = time.monotonic_ns()
        rows = [
            feedback(
                c["proposal"],
                m.requests[c["request_id"]].committed_token_ids,
                reject=cycle in (1, 2),
            )[0]
            for c in claims
        ]
        samples.append(
            dict(
                timestamp_ns=time.monotonic_ns(),
                requests=[
                    dict(
                        internal_request_id=c["request_id"],
                        actual_candidate_length=len(c["proposal"]["proposal_token_ids"]),
                    )
                    for c in claims
                ],
            )
        )
        m.feedback(rows)
        run_work(m)
        target_native.append(
            dict(
                host_start_ns=start,
                start_lower_ns=start,
                start_upper_ns=start,
                end_lower_ns=gpu_end,
                end_upper_ns=gpu_end,
                B=len(claims),
                gpu_event_ms=(gpu_end - start) / 1e6,
            )
        )
        runtime["target_steps"].append(
            dict(
                window=True,
                B=len(claims),
                start_ns=start,
                end_ns=time.monotonic_ns(),
                ping_admission=batch,
                rows=[
                    dict(
                        request_id=c["request_id"],
                        internal_request_id=c["request_id"],
                        candidate_positions=len(c["proposal"]["proposal_token_ids"]),
                        query_positions=len(c["proposal"]["proposal_token_ids"]) + 1,
                    )
                    for c in claims
                ],
            )
        )
    runtime["measurement_end_ns"] = time.monotonic_ns()
    native = [
        dict(
            purpose=f["purpose"],
            host_start_ns=f["start_ns"],
            B=f["B"],
            start_lower_ns=f["start_ns"],
            start_upper_ns=f["start_ns"],
            end_lower_ns=f["end_ns"],
            end_upper_ns=f["end_ns"],
            gpu_event_ms=0.1,
        )
        for f in m.backend.prepost_forwards.rows()
    ]
    cleanup(m)
    m.owner_stopped = True
    backend = {
        **m.backend.report(),
        "prepost": m.eager_report(),
        "fixed_device": {"forwards": native},
    }
    runtime["target_devices"] = [
        dict(
            device=dict(identity={"global_rank": r}, forwards=target_native),
            prepost_samples=samples.report(),
        )
        for r in (0, 1)
    ]
    # Synthetic native times above validate joins/union arithmetic, never GPU overlap.
    return runtime, backend


def test_collected_ping_events_report_rejects_loss_extra_recovery_and_false_ready():
    runtime, backend = collected()
    report = mechanism(runtime, backend)
    assert report["status"] == "COMPLETE", report["errors"]
    assert report["cross_home_admissions"] > 0
    assert report["generated"] == report["retained"] + report["discarded"]
    assert report["native_overlap"]["all"]["lower_ms"] > 0  # synthetic endpoints only
    for fault in ("lost", "missing", "legacy", "unfenced", "duplicate_claim"):
        b = copy.deepcopy(backend)
        if fault == "lost":
            b["prepost"]["pingpong"]["retention"]["phases"]["measurement"]["dropped_rows"] = 1
        elif fault == "missing":
            b["fixed_device"]["forwards"].pop()
        elif fault == "legacy":
            b["fixed_device"]["forwards"][0]["purpose"] = "proposal"
        elif fault == "unfenced":
            b["prepost_physical"]["forwards"][0]["worker_fenced"] = False
        else:
            events = b["prepost"]["pingpong"]["events"]
            events.append(next(e for e in events if e["event"] == "admission"))
        bad = mechanism(runtime, b)
        assert bad["status"] == "INCOMPLETE" and bad["errors"]
        if fault in ("missing", "legacy"):
            assert bad["native_overlap"]["all"] is None
        else:
            # Protocol diagnostic failure cannot erase independently intact device evidence.
            assert bad["native_overlap"] == report["native_overlap"]


def test_single_package_dedup_projection_and_failures_preserve_original(tmp_path, monkeypatch):
    runtime, backend = collected()
    directory = tmp_path / "delivery"
    for mode in ("pingpong-prepost3", "pingpong-eager-prepost3"):
        root = directory / "points" / mode
        root.mkdir(parents=True)
        (root / "runtime.json").write_text(
            json.dumps({**runtime, "unrelated_large_dump": [1] * 1000})
        )
        (root / "draft-backend-report.json").write_text(json.dumps(backend))
        (root / "config.json").write_text('{"shared":true}')
    (directory / "first-failure.json").write_text('{"first_exit_code":23}')
    archive = tmp_path / "delivery.tar.gz"
    result = export(directory, archive, first_code=23, stage="diagnostic_evidence")
    assert result["first_exit_code"] == 23 and result["missing"] > 0
    # This owner-only fixture has no worker snapshot. Export it, but never claim
    # the identity contract can be replayed; the producer-chain test covers COMPLETE.
    assert result["export_status"] == "INCOMPLETE"
    assert all("missing target_final_memory" in e for e in result["export_errors"])
    with tarfile.open(archive) as pack:
        inv = json.load(pack.extractfile("inventory.json"))
        paths = inv["logical_paths"]
        assert (
            paths["points/pingpong-prepost3/config.json"]
            == paths["points/pingpong-eager-prepost3/config.json"]
        )
        raw_runtime = json.load(pack.extractfile(paths["points/pingpong-prepost3/runtime.json"]))
        raw_backend = json.load(
            pack.extractfile(paths["points/pingpong-prepost3/draft-backend-report.json"])
        )
        assert "unrelated_large_dump" not in raw_runtime
        assert mechanism(raw_runtime, raw_backend) == mechanism(runtime, backend)
        assert len(pack.getnames()) == len(set(paths.values())) + 1
    assert "unrelated_large_dump" in json.loads(
        (directory / "points/pingpong-prepost3/runtime.json").read_text()
    )
    with pytest.raises(ValueError, match="new delivery"):
        export(directory, archive)
    monkeypatch.setattr("specrhythm.serving.ping_prepost_delivery.FILE_LIMIT", 1)
    omitted = export(directory, tmp_path / "omitted.tar.gz", first_code=23)
    assert omitted["export_status"] == "INCOMPLETE" and omitted["first_exit_code"] == 23


def test_control_report_counts_serial_extensions_and_joint_coverage_rejects_missing():
    from specrhythm.serving.ping_prepost_gpu_check import coverage

    runtime, backend = collected(eager=False)
    report = mechanism(runtime, backend)
    assert report["status"] == "COMPLETE", report["errors"]
    roles = report["per_role_window_forwards"]
    assert roles["normal_extension"]["count"] == 15
    assert roles["post"]["count"] == 5
    assert report["generated"] == report["retained"] == 0
    assert coverage(runtime, backend)["status"] == "INCOMPLETE"
    runtime, backend = collected()
    assert coverage(runtime, backend)["status"] == "COMPLETE"
    broken = copy.deepcopy(backend)
    broken["prepost"]["pingpong"]["claims"] = ["unfinished"]
    with pytest.raises(ValueError, match="release"):
        coverage(runtime, broken)


def test_malformed_point_failure_still_exports_original_bytes(tmp_path):
    directory = tmp_path / "delivery"
    point = directory / "points/pingpong-prepost3/runs/failed/point.json"
    point.parent.mkdir(parents=True)
    point.write_bytes(b'{"interrupted":')
    archive = tmp_path / "failure.tar.gz"
    status = export(directory, archive, first_code=23, stage="execution")
    assert status["export_status"] == "INCOMPLETE" and status["first_exit_code"] == 23
    with tarfile.open(archive) as pack:
        path = status["logical_paths"][str(point.relative_to(directory))]
        assert pack.extractfile(path).read() == point.read_bytes()
    assert any("invalid point metadata" in e for e in status["export_errors"])


def test_ping_warmup_allows_real_repeated_ids_and_preserves_full_active_limit():
    from specrhythm.serving.decode_scan_plan import options
    from specrhythm.serving.ping_prepost_window import PingPrePostWindow

    w = PingPrePostWindow(options(), 16, True)
    pop = dict(
        active_requests=16,
        request_ids=list(range(16)),
        cohorts={"A": list(range(8)), "B": list(range(8, 16))},
    )
    for i in range(4):
        assert not w.ready(100 + i * 10, population=pop)
        w.step_completed(
            dict(
                B=8,
                request_ids=list(range(8)),
                cohort="AB"[i % 2],
                start_ns=100 + i * 10,
                committed_tokens=8,
            ),
            105 + i * 10,
        )
    assert w.warmup_rotations == 2 and w.warmup_steps == 4
    assert not w.ready(200, population={**pop, "active_requests": 15})
    assert w.ready(210, population=pop)
    assert w.warmup_boundary()["committed_tokens_excluded"] == 32


def test_large_phased_ping_trace_exports_all_measurement_rows_and_declares_loss(
    tmp_path, monkeypatch
):
    from specrhythm.continuation.trace import PHASE_BUDGETS, TRACE

    monkeypatch.setattr(TRACE, "phase", "measurement")
    records = Records()
    for i in range(25001):
        records.append(dict(event="ready", request_id="a", prefix_version=i))
    report = records.report()
    assert len(report["rows"]) == 25001 and report["dropped_rows"] == 0
    directory = tmp_path / "delivery"
    directory.mkdir()
    path = directory / "draft-backend-report.json"
    path.write_text(
        json.dumps(
            {
                "prepost": {
                    "pingpong": {
                        "events": records.rows(),
                        "retention": {k: v for k, v in report.items() if k != "rows"},
                    }
                }
            }
        )
    )
    archive = tmp_path / "large.tar.gz"
    export(directory, archive, first_code=23)
    with tarfile.open(archive) as pack:
        inv = json.load(pack.extractfile("inventory.json"))
        data = json.load(pack.extractfile(inv["logical_paths"][path.name]))
        assert len(data["prepost"]["pingpong"]["events"]) == 25001
        assert data["prepost"]["pingpong"]["retention"]["dropped_rows"] == 0
    for i in range(PHASE_BUDGETS["measurement"] - 25001 + 1):
        records.append(dict(event="ready", request_id="a", prefix_version=i + 25001))
    assert len(records.rows()) == PHASE_BUDGETS["measurement"]
    assert records.report()["dropped_rows"] == 1
