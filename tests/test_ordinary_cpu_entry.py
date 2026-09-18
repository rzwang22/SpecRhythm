"""Production drive -> summarize -> archive -> reread; runner first failure and schema."""

import json
import subprocess
import tarfile

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results, dual_batch_run
from specrhythm.serving.common import read_json
from specrhythm.serving.k3 import B128
from specrhythm.serving.k3_acceptance import native_geometry
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("mode", ["serial-k3", "pingpong-k3"])
@pytest.mark.parametrize("stage", ["capacity_probe", "correctness", "performance"])
def test_shared_actual_drive_geometry_report_export_replay(driven, tmp_path, mode, stage):
    h = driven(
        mode,
        stage,
        full_run=True,
        configuration=B128,
        target_dispatch="shared-command",
        target_cpu="block-sets",
    )
    summarize = (
        decode_scan_results.summarize
        if stage != "correctness"
        else __import__("specrhythm.serving.fixed_results", fromlist=["summarize"]).summarize
    )
    result = summarize(h.path, h.directory, h.point, probe=h.probe)
    assert result["valid"], result
    assert h.runtime["target_pool_final"]["ownership_check"] == "block-sets"
    assert h.runtime["diagnostic_configuration"]["target_cpu"] == "block-sets"
    archive = tmp_path / "delivery.tar.gz"
    export(h.directory, archive, first_code=1, modes=(mode,))
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        replay = json.load(t.extractfile(paths["runtime.json"]))
    assert replay["target_pool_final"] == h.runtime["target_pool_final"]
    assert replay["diagnostic_configuration"] == h.runtime["diagnostic_configuration"]
    if stage == "performance":
        proof = native_geometry(replay, mode, configuration=B128)
        assert proof["full_batch_steps"] > 0
        assert len(proof["first_full_batch"]["request_ids"]) == (
            128 if mode == "serial-k3" else 64
        )


def test_cpu_runner_stops_first_error_and_single_package_preserves_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SR_FIXED_S1", "/missing")
    monkeypatch.setattr(
        subprocess, "check_output", lambda a, **kw: "a" * 40 if a[1] == "rev-parse" else b""
    )

    def fail(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(23, args)

    monkeypatch.setattr(subprocess, "run", fail)
    directory = tmp_path / "delivery"
    assert dual_batch_run.run(tmp_path, directory, "a" * 40, cpu_comparison=True) == 23
    assert len(calls) == 1
    plan = read_json(directory / "dual-batch-plan.json")
    assert plan["order"] == ["P0", "P1", "S1", "S1", "P1", "P0"]
    assert plan["cases"]["S1"]["mode"] == "serial-k3"
    assert plan["cases"]["P0"]["target_cpu"] == "reference"
    assert read_json(directory / "first-failure.json")["mode"] == "pingpong-k3"
    archive = tmp_path / "delivery.tar.gz"
    export(directory, archive, first_code=23, modes=())
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile("inventory.json"))
        assert inv["first_exit_code"] == 23
        out = json.load(t.extractfile(inv["logical_paths"]["runner-outcome.json"]))
        assert out["first_exit_code"] == 23
    assert not (directory / "smoke").exists() and not (directory / "windows").exists()


def test_cpu_policy_missing_actual_audit_is_not_metadata_pass():
    from specrhythm.serving.target_dispatch import qualification_errors

    v = {
        "cycle_accounting": {"matched_step_summary": {"complete": 1}, "request_cycle_summary": {}}
    }
    opts = {"target_dispatch": "shared-command", "target_cpu": "block-sets"}
    runtime = {"diagnostic_configuration": dict(opts)}
    assert any("audit" in e for e in qualification_errors(v, runtime, opts))
    runtime["target_pool_final"] = {"ownership_check": "reference"}
    assert any("audit" in e for e in qualification_errors(v, runtime, opts))
    runtime["target_pool_final"]["ownership_check"] = "block-sets"
    assert qualification_errors(v, runtime, opts) == []


def test_runner_reaches_exact_six_case_order_without_reference_performance(tmp_path, monkeypatch):
    """Orchestration test; actual report/qualifier is covered by drive tests above."""
    from pathlib import Path

    from specrhythm.serving import (
        fixed_artifacts,
        k3_acceptance,
        ordinary_cpu_report,
        ping_prepost_delivery,
    )

    monkeypatch.setenv("SR_FIXED_S1", "/frozen")
    monkeypatch.setattr(
        subprocess, "check_output", lambda a, **kw: "a" * 40 if a[1] == "rev-parse" else b""
    )
    commands = []
    modes = {}

    def command(args, **kw):
        commands.append(args)
        root = Path(args[args.index("--root") + 1])
        if "prepare" in args:
            root.mkdir(parents=True)
            cpu = args[args.index("--target-cpu") + 1]
            dispatch = args[args.index("--target-dispatch") + 1]
            (root / "scan-config.json").write_text(
                json.dumps(
                    dict(
                        selection=[1, 2],
                        workload_sha256="cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4",
                        execution={"same": True},
                        options={"target_cpu": cpu, "target_dispatch": dispatch},
                    )
                )
            )
        if "run" in args:
            modes[str(root)] = args[args.index("--mode") + 1]
            (root / "runtime.json").write_text("{}")
            (root / "draft-backend-report.json").write_text("{}")

    monkeypatch.setattr(subprocess, "run", command)
    smoke_calls = []

    def smoke(source, root, dispatch, **kw):
        smoke_calls.append((root.name, kw["mode"], dispatch, kw["target_cpu"]))
        return dict(
            target_dispatch=dispatch,
            run_directory=str(root),
            outputs={"r": {"tokens": [1]}},
            mode=kw["mode"],
        )

    monkeypatch.setattr(dual_batch_run, "smoke", smoke)
    monkeypatch.setattr(
        fixed_artifacts,
        "point_reports",
        lambda root: [dict(point={"probe": False}, artifact=str(root))],
    )
    checks = []
    monkeypatch.setattr(
        k3_acceptance, "measurement", lambda report, raw, mode, *a, **kw: checks.append(mode)
    )
    monkeypatch.setattr(
        ping_prepost_delivery,
        "comparison",
        lambda case, **kw: dict(
            valid=True,
            points=[dict(committed_tokens=100, window_ms=30_000, throughput_tok_s=100 / 30)],
        ),
    )
    monkeypatch.setattr(
        ordinary_cpu_report, "window_metrics", lambda raw: dict(natural_completions_in_window=0)
    )
    monkeypatch.setattr(
        ordinary_cpu_report,
        "recovery_during_prepare",
        lambda *a: dict(status="CPU_ORCHESTRATION_ONLY"),
    )
    directory = tmp_path / "delivery"
    assert dual_batch_run.run(tmp_path, directory, "a" * 40, cpu_comparison=True) == 0
    assert checks == [
        "pingpong-k3",
        "pingpong-k3",
        "serial-k3",
        "serial-k3",
        "pingpong-k3",
        "pingpong-k3",
    ]
    assert [x[0] for x in smoke_calls] == ["P0", "P1", "S1", "S0"]
    assert [Path(c[c.index("--root") + 1]).parents[1].name for c in commands if "run" in c] == [
        "0-P0",
        "1-P1",
        "2-S1",
        "3-S1",
        "4-P1",
        "5-P0",
    ]
    result = read_json(directory / "comparison.json")
    assert result["valid"] and set(result["aggregate"]) == {"P0", "P1", "S1"}
    assert result["output_equivalence_status"] == "NOT_RUN"
