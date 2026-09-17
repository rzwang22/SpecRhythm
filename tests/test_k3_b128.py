"""B128 production capacity/drive/native hooks/qualification/export, without GPUs."""

import copy
import json
import re
import tarfile
from pathlib import Path

import pytest
from test_k3_acceptance import entry, retain
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results
from specrhythm.serving.common import DataError
from specrhythm.serving.decode_scan_plan import manifest, options, selected_point
from specrhythm.serving.k3 import B16, B64, B128, MODES, geometry, validate_point
from specrhythm.serving.k3_acceptance import full_batch_receipt, native_geometry
from specrhythm.serving.k3_capacity import qualify
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ["capacity_probe", "performance"])
def test_b128_real_drive_to_summary_export_and_entry(mode, stage, driven, tmp_path):
    h = driven(
        mode,
        stage,
        full_run=True,
        configuration=B128,
        validation_profile="performance-exploration",
    )
    result = decode_scan_results.summarize(h.path, h.directory, h.point, probe=h.probe)
    assert result["valid"], result.get("primary_error")
    assert result["output_equivalence_status"] == "NOT_RUN"
    assert h.runtime["point"]["batch"] == 128
    assert qualify(h.actual, mode)["status"] == "PASS"
    if not h.probe:
        proof = native_geometry(h.runtime, mode, full_fixture=True, configuration=B128)
        assert full_batch_receipt(proof, mode, B128)
        assert not full_batch_receipt(proof, mode, B64)
        assert len(proof["first_full_batch"]["request_ids"]) == (128 if mode in MODES[:2] else 64)
        assert set(proof["observed_home_cohorts"]) == set(geometry(mode, B128)["home_capacities"])
        root, result = retain(h)
        # Actual CLI summary also derives its ceiling from the selected geometry.
        from specrhythm.serving.decode_scan_cli import summary
        from specrhythm.serving.decode_scan_plan import SCHEMA
        from specrhythm.serving.s2_plan import sealed

        (root / "scan-config.json").write_text(json.dumps(sealed(dict(
            schema_version=SCHEMA, k3_configuration=B128,
            validation_profile="performance-exploration", pool_size=360,
            workload_sha256=result["workload_sha256"],
            execution={"git_commit": result["git_commit"]},
            options=json.loads(h.path.read_text())["fixed_diagnostic"]["options"],
            points=[], optional_points=[h.point]))))
        rows = summary(root)["points"]
        assert len(rows) == 1 and rows[0]["sub_batch"] == geometry(mode, B128)[
            "target_request_ceiling"]
        boundary = result["scan_warmup_boundary"]
        assert boundary["request_opportunities"] >= 256 and boundary["unique_requests"] == 128
        assert result["request_verification_opportunities"] == sum(
            s["B"] for s in h.runtime["target_steps"] if s["window"]
        )
        body = re.search(
            r"<<'PY_MEASUREMENT'\n(.*?)\nPY_MEASUREMENT",
            Path("scripts/run_k3_b128.sh").read_text(),
            re.S,
        )[1]
        accepted = entry(root, mode, body, validation_profile="performance-exploration")
        assert accepted.returncode == 0, accepted.stderr
    archive = tmp_path / "one.tar.gz"
    export(h.directory, archive, first_code=1, modes=MODES)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        runtime = json.load(t.extractfile(paths["runtime.json"]))
        actual = json.load(t.extractfile(paths["actual-capacity.json"]))
    assert qualify(actual, mode) == qualify(h.actual, mode)
    assert (
        runtime["probe"] is h.probe and runtime["validation_profile"] == "performance-exploration"
    )
    if not h.probe:
        assert native_geometry(runtime, mode, full_fixture=True) == proof


@pytest.mark.parametrize("mode", MODES)
def test_b128_configuration_manifest_and_conservative_capacity(mode):
    from types import SimpleNamespace as NS

    from specrhythm.serving.k3_capacity import check, preflight

    assert preflight(B128)["GPU_capacity"] == "PENDING"
    g = geometry(mode, B128)
    assert g["active_limit"] == g["draft_physical_batch_ceiling"] == 128
    assert g["home_capacities"] == ({"A": 128} if mode in MODES[:2] else {"A": 64, "B": 64})
    assert geometry(mode, B16)["active_limit"] == 16
    assert geometry(mode, B64)["active_limit"] == 64
    m = manifest(
        {},
        [str(i) for i in range(360)],
        "sha",
        options(),
        128,
        k3_configuration=B128,
        validation_profile="performance-exploration",
    )
    point = selected_point(
        mode, 128, k3_configuration=B128, validation_profile="performance-exploration"
    )
    validate_point(point, m)
    assert m["fixed_diagnostic"]["initial_request_ids"] == [str(i) for i in range(128)]
    assert m["fixed_diagnostic"]["diagnostic_configuration"]["capture_algorithm_changed"] is False
    meta = m["fixed_diagnostic"]["capacity"][mode]
    defs = [NS(request_id=str(i), prompt_length=15, maximum_new_tokens=100) for i in range(360)]
    for role, gpu in [("target", 1), ("draft", 0)]:
        rank = dict(
            role=role,
            mode=mode,
            physical_gpu_id=gpu,
            gpu_uuid=f"GPU-{gpu}",
            block_size=16,
            num_gpu_blocks=10000,
            vocab_size=100,
            free_memory_bytes=2**30,
        )
        good = check(defs, rank, mode=mode, active_limit=128, metadata=meta)
        assert good["valid"] and good["active_limit"] == 128
        assert good["required_speculative_positions"] == (
            6 if role == "draft" and g["eager"] else 3
        )
        assert good["reserved_speculative_positions"] == (
            6 if role == "draft" and g["eager"] else 4
        )
        assert not check(
            defs, dict(rank, num_gpu_blocks=1), mode=mode, active_limit=128, metadata=meta
        )["valid"]
        with pytest.raises(DataError, match="batch/query"):
            check(
                defs,
                rank,
                mode=mode,
                active_limit=128,
                metadata={**meta, role + "_query_token_limit": 63},
            )
    with pytest.raises(DataError):
        validate_point({**point, "batch": 64}, m)
    with pytest.raises(DataError):
        validate_point({**point, "k3_configuration": B64}, m)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate",
        "duplicate_forward",
        "small_engine",
        "query_capacity",
        "wrong_geometry",
        "wrong_top_configuration",
        "active_overflow",
        "home_overflow",
    ],
)
def test_b128_raw_failures_not_report_labels(fault, driven):
    h = driven(MODES[0], "performance", full_run=True, configuration=B128)
    r, a = copy.deepcopy(h.runtime), copy.deepcopy(h.actual)
    if fault in ("small_engine", "query_capacity"):
        key = "max_num_seqs" if fault == "small_engine" else "max_num_batched_tokens"
        for row in a["target_effective_by_rank"]:
            row[key] = 64 if fault == "small_engine" else 511
        for row in a["target_worker_ranks"]:
            row["s1_effective_capacity"][key] = 64 if fault == "small_engine" else 511
        # Raw worker equality remains true: actual insufficient capacity.
        with pytest.raises(DataError, match="insufficient"):
            qualify(a, MODES[0])
        return
    if fault == "missing_rank":
        r["target_devices"].pop()
    elif fault == "wrong_top_configuration":
        r["k3_configuration"] = B64
    elif fault == "wrong_geometry":
        r["capacity"]["execution_geometry"] = geometry(MODES[0], B64)
    elif fault in ("active_overflow", "home_overflow"):
        pop = r["target_steps"][0]["population"]
        if fault == "active_overflow":
            pop["active_requests"] = 129
        else:
            pop["cohort_held"]["A"] = 129
    else:
        f = r["target_devices"][0]["device"]["forwards"][0]
        if fault == "duplicate":
            f["internal_request_ids"][0] = f["internal_request_ids"][1]
        else:
            r["target_devices"][0]["device"]["forwards"].append(copy.deepcopy(f))
    with pytest.raises(DataError):
        native_geometry(r, MODES[0], full_fixture=True, configuration=B128)


@pytest.mark.parametrize(
    "mode,limit", [(MODES[0], 64), (MODES[1], 32), (MODES[2], 32), (MODES[3], 16)]
)
def test_b128_multiple_small_forwards_cannot_fake_native_full_batch(mode, limit, driven):
    h = driven(mode, "correctness", full_run=True, configuration=B128, admission_limit=limit)
    with pytest.raises(DataError, match="lacks one native Target forward"):
        native_geometry(h.runtime, mode, full_fixture=True, configuration=B128)


def test_b128_warmup_counts_actual_opportunities_and_requires_all_active():
    from specrhythm.serving.k3_window import K3Window

    w = K3Window(dict(warmup_steps=2, window_seconds=30), 128, MODES[2], B128)
    pop = dict(request_ids=[str(i) for i in range(128)])
    # 256 opportunities on only the first half must not complete warmup.
    for i in range(4):
        w.step_completed(
            dict(
                B=64,
                request_ids=[str(j) for j in range(64)],
                cohort="A",
                start_ns=i * 2 + 1,
                committed_tokens=64, rows=[],
            ),
            i * 2 + 2,
        )
    assert not w.warmup_satisfied(pop)
    for i in range(8):
        w.step_completed(
            dict(
                B=8,
                request_ids=[str(j) for j in range(64 + i * 8, 72 + i * 8)],
                cohort="B",
                start_ns=20 + i * 2,
                committed_tokens=8, rows=[],
            ),
            21 + i * 2,
        )
    assert w.warmup_satisfied(pop)
    assert w.warmup_boundary()["request_opportunities"] == 320
    assert w.warmup_boundary()["unique_requests"] == 128


def test_capture_summary_clips_and_unions_per_thread_without_gpu_arithmetic():
    from specrhythm.serving.k3_scale_report import capture_summary

    rows = [
        dict(
            category="target_forward_diagnostics",
            pid=1,
            thread_id=2,
            start_ns=a * 10**6,
            end_ns=b * 10**6,
        )
        for a, b in [(-1, 3), (2, 7), (8, 12)]
    ]
    r = capture_summary({"rank0": {"intervals": rows}}, 0, 10**7)["rank0"]
    assert r["count"] == 3 and r["inclusive_sum_ms"] == 10
    assert r["mean_ms"] == 10 / 3 and r["p50_ms"] == 3 and r["p95_ms"] == 5
    assert r["threads"][0]["union_ms"] == 9 and r["threads"][0]["window_share"] == 0.9
    assert capture_summary({"rank1": {}}, 0, 10**7)["rank1"]["mean_ms"] is None


def test_explicit_capacity_interface_cli_b128_before_any_model(tmp_path):
    import os
    import subprocess
    import sys

    destination = tmp_path / 'static.json'
    r = subprocess.run([sys.executable, '-m', 'specrhythm.serving.k3_capacity',
                        '--k3-configuration', B128, '--output', str(destination)],
                       capture_output=True, text=True, timeout=20,
                       env={**os.environ, 'PYTHONPATH': str(Path('src').resolve())})
    assert r.returncode == 0, r.stderr
    value = json.loads(destination.read_text())
    assert value['GPU_capacity'] == 'PENDING'
    assert {r['geometry']['active_limit'] for r in value['reservations']} == {128}


def test_b128_repeated_plan_runs_only_four_initial_probes(tmp_path, monkeypatch):
    from specrhythm.phase4.manifest import atomic_write_json
    from specrhythm.serving import k3_repeat_run as repeat

    calls = []

    def child(argv, env):
        calls.append((argv, env))
        root = Path(env['SR_K3_LOCAL_DELIVERY'])
        root.mkdir()
        atomic_write_json(root / 'runner-outcome.json', dict(
            first_exit_code=23 if len(calls) == 2 else 0, stage='execution', mode=MODES[-1]))
        return 23 if len(calls) == 2 else 0

    monkeypatch.setattr(repeat.subprocess, 'call', child)  # External GPU child only.
    root = tmp_path / 'delivery'
    assert repeat.run(root, tmp_path, 'a'*40, 'unified', B128) == 23
    assert [e['SR_K3_SKIP_CAPACITY'] for _, e in calls] == ['0', '1']
    assert [e['SR_K3_REVERSE'] for _, e in calls] == ['0', '1']
    assert all(a[1].endswith('run_k3_b128.sh') for a, _ in calls)
    assert all(e['SR_K3_DRAFT_DISPATCH'] == 'unified' and
               e['SR_K3_OBSERVATION'] == 'deferred-window' for _,e in calls)
    plan = repeat.read_declaration(root)
    assert plan['k3_configuration'] == B128 and plan['output_equivalence_status'] == 'NOT_RUN'
    assert plan['repeats'][0]['modes'] == list(MODES)
    assert plan['repeats'][1]['modes'] == list(reversed(MODES))


def test_reverse_b128_child_reuses_probes_and_runs_all_modes(tmp_path, monkeypatch):
    import test_k3_runbook as prior

    monkeypatch.setenv('SR_K3_REVERSE', '1')
    monkeypatch.setenv('SR_K3_SKIP_CAPACITY', '1')
    monkeypatch.setattr(prior, 'MODES', tuple(reversed(MODES)))
    prior.test_one_bundle_first_error_stops_points_and_parent_remains_open(
        tmp_path, 'none', '', False, 'run_k3_b128.sh')
    calls = (tmp_path / 'calls.txt').read_text().splitlines()
    assert not any(c.endswith((' capacity', ' correctness')) for c in calls)
    assert [c for c in calls if c.endswith(' run')] == [m+' run' for m in reversed(MODES)]
