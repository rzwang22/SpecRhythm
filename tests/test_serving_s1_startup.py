"""CPU-only S1 adapter -> real Serial runner -> simulated LLM construction contracts."""

import os
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_serving_s1 import execution, freeze_serial_inputs
from test_serving_s1 import phase4_config as _phase4_config

from specrhythm.phase4.decode_ready import DecodeReadyProvenance
from specrhythm.phase4.manifest import sha256_file
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.s1_policy import REQUIRED_ENVIRONMENT
from specrhythm.serving.s1_workload import load_execution, write_once

phase4_config = _phase4_config


class WorkerCreationReached(RuntimeError):
    pass


@pytest.fixture
def serial_startup(tmp_path, monkeypatch, phase4_config):
    from specrhythm.phase4 import serial_runner
    from specrhythm.serving import s1_runtime

    root = tmp_path / "root"
    path, manifest, _ = execution(root / "profile")
    freeze_serial_inputs(root, path, manifest)
    directory = root / "G1/0-serial/attempt-001"
    directory.mkdir(parents=True)
    for name in ("environment.json", "topology.json"):
        write_once(root / name, {})
    write_once(
        directory / "draft-service-ready.json",
        dict(
            schema_version="specrhythm.phase4-draft-service-ready.v1",
            provenance={},
        ),
    )
    # Only unavailable S0 parent/environment/GPU installation are substituted.
    # Adapter, config/patch loading, workload checks, Serial runner and builder stay real.
    monkeypatch.setattr(s1_runtime, "load_execution", lambda p, **kw: load_execution(p))
    for name in ("validate_environment", "validate_topology"):
        monkeypatch.setattr(serial_runner, name, lambda *a: {"valid": True, "errors": []})
    for name in ("validate_installed_patched_runner", "validate_installed_patch_stack"):
        monkeypatch.setattr(serial_runner, name, lambda *a: {})
    for key, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    for key in ("USE_TORCH", "USE_TF", "USE_FLAX", "VLLM_BATCH_INVARIANT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    entered = []

    class LLM:
        def __init__(self, **kwargs):
            context_path = Path(os.environ["SR_PHASE4_DECODE_READY_CONTEXT"])
            assert context_path == directory / "decode-ready-context.json"
            assert context_path.is_file(), "Serial context missing at LLM construction"
            context = read_json(context_path)
            provenance = DecodeReadyProvenance.from_dict(context)
            assert provenance.specrhythm_git_commit == manifest["execution"]["git_commit"]
            assert provenance.workload_sha256 == manifest["logical"]["workload_sha256"]
            assert provenance.target_model_path == str(phase4_config.target.resolved_model_path)
            assert provenance.draft_model_path == str(phase4_config.draft.resolved_model_path)
            assert provenance.target_physical_gpu_ids == (1, 2)
            assert provenance.draft_physical_gpu_ids == (0,)
            assert (
                provenance.target_tensor_parallel_size,
                provenance.draft_tensor_parallel_size,
            ) == (2, 1)
            assert provenance.sampling_configuration == phase4_config.sampling.to_dict()
            assert (
                provenance.batch_invariant_configuration["correctness_mode"] == "batch-invariant"
            )
            patch = read_json(root / "patch-manifest.json")
            assert provenance.vllm_patch_stack_sha256 == tuple(
                r["patch_sha256"] for r in patch["patch_stack"]
            )
            binding = context["s1_execution_binding"]
            assert binding["execution_sha256"] == manifest["manifest_sha256"]
            assert binding["execution_configuration"] == manifest["execution"]
            assert binding["config_sha256"] == sha256_file(root / "config.json")
            assert binding["patch_manifest_sha256"] == sha256_file(root / "patch-manifest.json")
            assert kwargs["speculative_config"]["model"].endswith(".RemoteDraftProposer")
            entered.append(provenance)
            raise WorkerCreationReached("original simulated LLM startup exception")

    configure = serial_runner.configure_before_worker_creation

    def configure_then_expose_vllm(mode):
        evidence = configure(mode)
        monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=object))
        return evidence

    monkeypatch.setattr(
        serial_runner, "configure_before_worker_creation", configure_then_expose_vllm
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    with tempfile.TemporaryDirectory(prefix="sr-s1-order-", dir="/tmp") as sockets:
        with socket.socket(socket.AF_UNIX) as sock:
            socket_path = Path(sockets) / "draft.sock"
            sock.bind(str(socket_path))
            monkeypatch.setenv("SR_S1_DRAFT_SOCKET", str(socket_path))
            yield root, path, directory, manifest, entered


def test_serial_adapter_creates_and_parses_context_before_real_runner_llm(serial_startup):
    from specrhythm.serving.s1_runtime import run_consumer

    root, path, directory, _, entered = serial_startup
    context = directory / "decode-ready-context.json"
    assert not context.exists()
    with pytest.raises(WorkerCreationReached, match="original simulated"):
        run_consumer("serial", path, directory, root)
    assert len(entered) == 1
    before = context.read_bytes()
    # Only the adapter owns creation; repeat startup cannot overwrite retained evidence.
    with pytest.raises(FileExistsError, match="context"):
        run_consumer("serial", path, directory, root)
    assert len(entered) == 1 and context.read_bytes() == before


@pytest.mark.parametrize("name", ["config.json", "patch-manifest.json"])
def test_serial_context_rejects_changed_frozen_inputs_before_llm(serial_startup, name):
    from specrhythm.serving.s1_runtime import run_consumer

    root, path, directory, _, entered = serial_startup
    with (root / name).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="frozen Serial context input changed"):
        run_consumer("serial", path, directory, root)
    assert not entered and not (directory / "decode-ready-context.json").exists()


def test_serial_startup_exception_remains_primary_with_missing_reports(
    serial_startup, monkeypatch, capsys
):
    from specrhythm.phase4.manifest import atomic_write_json
    from specrhythm.serving import s1_cli
    from specrhythm.serving.s1_results import seal_run

    root, path, directory, manifest, entered = serial_startup
    monkeypatch.setattr(s1_cli, "validate_root_policy", lambda *a: None)
    assert (
        s1_cli.main(
            [
                "child",
                "--root",
                str(root),
                "--mode",
                "serial",
                "--manifest",
                str(path),
                "--directory",
                str(directory),
            ]
        )
        == 1
    )
    assert len(entered) == 1
    child_failure = read_json(directory / "child-failure.json")
    assert child_failure["error_type"] == "WorkerCreationReached"
    assert child_failure["error"] == "original simulated LLM startup exception"
    assert "run_serial_disaggregated" in child_failure["traceback"]
    original = (directory / "child-failure.json").read_bytes()

    def changed_input(*a):
        raise FileNotFoundError("secondary missing frozen input")

    monkeypatch.setattr(s1_cli, "validate_execution_files", changed_input)
    report = s1_cli.failed_execution_report(root, "serial", path, manifest, directory, 7)
    assert report["valid"] is report["performance_result"] is False
    assert report["primary_error"]["error"] == child_failure["error"]
    assert len(report["errors"]) == 1 and child_failure["error"] in report["errors"][0]
    assert report["error_details"]["returncode"] == 7
    assert report["cross_run_output_comparison"]["status"] == "NOT_REQUIRED"
    missing = [
        d
        for d in report["secondary_diagnostics"]
        if d.get("status") == "MISSING_AFTER_EXECUTION_FAILURE"
    ]
    assert {Path(d["artifact"]).name for d in missing} == {
        "raw.json",
        "plugin-report.json",
        "draft-backend-report.json",
    }
    assert any(
        d.get("error") == "secondary missing frozen input" for d in report["secondary_diagnostics"]
    )
    seal_run(directory, report)

    def failed_gate(*a):
        atomic_write_json(root / "stage.json", dict(mode="serial", directory=str(directory)))
        raise DataError(report["errors"][0], returncode=7, directory=str(directory))

    monkeypatch.setattr(s1_cli, "gate", failed_gate)
    assert s1_cli.supervise(root, "G1", "cpu-startup-failure") == 7
    output = capsys.readouterr().err
    assert child_failure["error"] in output
    assert "[S1-P serial secondary diagnostic]" in output
    assert "MISSING_AFTER_EXECUTION_FAILURE" in output
    assert (directory / "child-failure.json").read_bytes() == original
