"""Collect real owner/physical records, report them, reject incomplete mechanism evidence."""

from __future__ import annotations

import copy
import time

import pytest
from test_prepost_protocol import cleanup, feedback, machine
from test_serial_eager_owner import proposal_row, verify_row
from test_serving_s1 import request

from specrhythm.serving.prepost_evidence import analyze
from specrhythm.serving.prepost_gpu_check import compare_outputs, prepare
from specrhythm.serving.s2_plan import sealed


def collected():
    m = machine()
    runtime = dict(point={"mode": "serial-eager-prepost3"}, target_steps=[])
    target_samples = []
    native = []
    budgets = dict(a=100, b=100)
    prefixes = dict(a=(10, 20), b=(10, 20))
    proposals = m.batch_propose([proposal_row(rid) for rid in prefixes])["proposals"]
    runtime["measurement_start_ns"] = time.monotonic_ns()
    for cycle in range(3):
        start = time.monotonic_ns()
        m.verify_start([verify_row(p) for p in proposals])
        while m.step():
            pass
        rows, samples = [], []
        for p in proposals:
            rid = p["request_id"]
            row, prefixes[rid] = feedback(p, prefixes[rid], reject=rid == "b" and cycle == 0)
            rows.append(row)
            samples.append(
                dict(internal_request_id=rid, actual_candidate_length=len(p["proposal_token_ids"]))
            )
            budgets[rid] -= len(row["committed_delta"])
        target_samples.append(dict(timestamp_ns=time.monotonic_ns(), requests=samples))
        m.prepare_synchronizations(rows)
        m.finish_synchronizations(rows)
        runtime["target_steps"].append(
            dict(
                start_ns=start,
                end_ns=time.monotonic_ns(),
                window=True,
                B=2,
                committed_tokens=sum(len(r["committed_delta"]) for r in rows),
                rows=[
                    dict(
                        internal_request_id=p["request_id"],
                        candidate_positions=len(p["proposal_token_ids"]),
                        query_positions=len(p["proposal_token_ids"]) + 1,
                    )
                    for p in proposals
                ],
            )
        )
        proposals = m.batch_propose(
            [proposal_row(rid, prefixes[rid], cycle + 1, budgets[rid]) for rid in prefixes]
        )["proposals"]
    runtime["measurement_end_ns"] = time.monotonic_ns()
    for f in m.backend.prepost_forwards.rows():
        native.append(
            dict(purpose=f["purpose"], host_start_ns=f["start_ns"], B=f["B"], gpu_event_ms=0.1)
        )
    cleanup(m)
    m.owner_stopped = True
    report = m.backend.report()
    report["prepost"] = m.eager_report()
    report["fixed_device"] = {"forwards": native}  # Synthetic device numbers: only test joins.
    retention = report["prepost"]["retention"]
    runtime["target_devices"] = [
        dict(
            device={"identity": {"global_rank": rank}},
            prepost_samples={**retention, "rows": target_samples},
        )
        for rank in (0, 1)
    ]
    return runtime, report


def test_collected_records_pass_and_missing_or_extra_forwards_fail():
    runtime, backend = collected()
    result = analyze(runtime, backend)
    assert result["status"] == "COMPLETE", result["errors"]
    assert result["mixed_target_steps"] == 1
    assert all(c["forward_counts"]["prepost_post"] == 1 for c in result["cycles"])
    for fault in ("missing", "extra", "truncated", "bonus"):
        damaged = copy.deepcopy(backend)
        if fault == "missing":
            damaged["prepost"]["cycles"].pop(0)
        elif fault == "extra":
            f = damaged["fixed_device"]["forwards"][0].copy()
            f["purpose"] = "proposal"
            damaged["fixed_device"]["forwards"].append(f)
        elif fault == "bonus":
            damaged["prepost"]["cycles"][0]["requests"][0]["bonus"] = 1
        else:
            damaged["prepost"]["retention"]["phases"]["measurement"]["dropped_rows"] = 1
        assert analyze(runtime, damaged)["status"] == "FAILED"


def test_joint_fixture_is_separate_and_complete_outputs_required(tmp_path):
    import json

    from specrhythm.continuation.prepost import MODES

    source = tmp_path / "source"
    (source / "inputs").mkdir(parents=True)
    for name in ("config.json", "patch-manifest.json", "environment.json", "topology.json"):
        (source / name).write_text("{}")
    rows = [request(i, 100).to_dict() for i in range(16)]
    (source / "inputs/requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    trace = sealed(
        dict(rows=[dict(request_id=r["request_id"], arrival_offset_seconds=0) for r in rows])
    )
    (source / "inputs/execution-B16.json").write_text(
        json.dumps(
            sealed(
                dict(
                    execution={"capacity": {}},
                    trace=trace,
                    workload_sha256="f" * 64,
                    fixed_diagnostic={"options": {}},
                )
            )
        )
    )
    path = prepare(source, tmp_path / "fixture", MODES[1])
    value = json.loads(path.read_text())
    assert (
        value["actual_N"] == 16 and value["prepost_correctness_fixture"]["max_output_tokens"] == 32
    )
    assert all(
        json.loads(line)["maximum_new_tokens"] == 100
        for line in (source / "inputs/requests.jsonl").read_text().splitlines()
    )
    runtime = dict(
        stop_reason="all_naturally_completed",
        requests=[
            dict(
                request_id=str(i),
                state="FINISHED",
                resources_released=True,
                generated_token_ids=[1, 2, 3],
            )
            for i in range(16)
        ],
        target_steps=[dict(rows=[dict(candidate_positions=n) for n in (1, 4)])],
    )
    results = {mode: copy.deepcopy(runtime) for mode in ("target", *MODES)}
    assert compare_outputs(results)["valid"]
    results[MODES[1]]["requests"][0]["generated_token_ids"][-1] = 7
    with pytest.raises(ValueError, match="full output mismatch"):
        compare_outputs(results)


def test_joint_export_inventory_preserves_missing_and_hashes_raw_once(tmp_path):
    import hashlib
    import json
    import tarfile

    from specrhythm.serving.prepost_bundle import export

    root = tmp_path / "root"
    joint = root / "prepost-joint-gpu-check"
    joint.mkdir(parents=True)
    content = b'{"GPU_correctness":"FAILED"}'
    (joint / "failure.json").write_bytes(content)
    result = export(root, tmp_path / "evidence.tar.gz")
    with tarfile.open(result["joint_correctness"]["path"]) as archive:
        assert archive.getnames().count("failure.json") == 1
        assert archive.extractfile("failure.json").read() == content
        inventory = json.load(archive.extractfile("joint-inventory.json"))
    assert inventory["missing"] == 4
    assert inventory["inventory"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert not (joint / "result.json").exists()


def test_joint_execution_failure_retains_primary_error_if_recording_fails(tmp_path, monkeypatch):
    from specrhythm.serving import fixed_cli
    from specrhythm.serving import prepost_gpu_check as check
    from specrhythm.serving.common import DataError

    source = tmp_path / "point"
    source.mkdir()
    monkeypatch.setattr(check, "prepare", lambda *args: tmp_path / "manifest.json")
    primary = DataError("owned execution failed", returncode=23)

    def failed(*args, **kwargs):
        raise primary

    def unwritable(*args, **kwargs):
        raise OSError("record write failed")

    monkeypatch.setattr(fixed_cli, "run_point", failed)
    monkeypatch.setattr(check, "write_once", unwritable)
    with pytest.raises(DataError) as error:
        check.run(source)
    assert error.value is primary and error.value.details["returncode"] == 23

    monkeypatch.setattr(check, "run", failed)
    with pytest.raises(SystemExit) as exit_result:
        check.main(["--root", str(source)])
    assert exit_result.value.code == 23
