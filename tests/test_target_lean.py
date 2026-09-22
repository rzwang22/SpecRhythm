"""Real capture, sampler, report and archive contracts; only GPU tensors are substituted."""

import copy
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_phase4_vllm_diagnostics import _capture, _serial_proposal
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced
from test_prepost_target import Array, make_runner
from test_prepost_target import runner_class as _runner_class

from specrhythm.phase4 import target_profile, vllm_diagnostics
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.serving import decode_scan_results, fixed_observe
from specrhythm.serving.common import DataError
from specrhythm.serving.fixed_identity import BoundPromptIdentityMap
from specrhythm.serving.k3 import B128, MODES
from specrhythm.serving.k3_scale_report import diagnostic_substages
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.runtime_profile import load_runtime_requests

hardware, produced, driven, runner_class = _hardware, _produced, _driven, _runner_class


@pytest.fixture(autouse=True)
def isolated_input_environment(monkeypatch):
    # Other production-configuration fixtures intentionally call os.environ.update.
    # This module creates its OWN tiny or S2 workload; never inherit another test's
    # serving manifest/numerical plan. The real producer installs its own S2 paths.
    for key in ("SR_S1_EXECUTION_MANIFEST", "SR_S2_EXECUTION_MANIFEST",
                "SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN", "SR_FIXED_TARGET_DIAGNOSTICS"):
        monkeypatch.delenv(key, raising=False)


def initialize(runner, logits):
    import os

    path = Path(os.environ["SR_PHASE4_WORKLOAD"])
    definitions = load_runtime_requests(path, expected_count=5, require_task_mixture=False)
    runner.drafter.identity = BoundPromptIdentityMap(
        FrozenPromptIdentityMap.from_definitions(definitions))
    target_profile.initialize(runner)


@pytest.mark.parametrize("selected", ["full", "lean"])
def test_real_capture_collection_validator_without_optional_sampler_dependency(
    selected, tmp_path, monkeypatch
):
    monkeypatch.setenv(target_profile.ENV, selected)
    monkeypatch.setattr(fixed_observe, "TARGET_ROWS", [])
    original = CheckpointJsonl.append

    def append(log, row):
        fixed_observe.capture(row)
        return original(log, row)

    monkeypatch.setattr(CheckpointJsonl, "append", append)
    setup_state = {}

    def setup(runner, logits):
        initialize(runner, logits)
        setup_state.update(runner=runner, logits=logits)
        if selected == "lean":
            def forbidden(*args, **kwargs):
                pytest.fail("lean attempted optional logits computation/copy or workload reload")

            monkeypatch.setattr(logits, "detach", forbidden)
            monkeypatch.setattr(logits, "log_softmax", forbidden)
            monkeypatch.setattr(vllm_diagnostics, "capture_target_numerical_rows", forbidden)
            monkeypatch.setattr(vllm_diagnostics, "load_smoke_requests", forbidden)

    row = _capture(tmp_path, monkeypatch, _serial_proposal(), setup=setup)
    assert vllm_diagnostics.validate_runtime_target_diagnostic(row) == []
    assert fixed_observe.TARGET_ROWS[0]["structural_errors"] == []
    assert row["proposal_token_ids"] == [10, 11]
    assert row["position_ids"] == [3, 4, 5]
    if selected == "lean":
        assert row["numerical_forensics"] == target_profile.NOT_COLLECTED
        assert not any(k in row for k in target_profile.NUMERICAL_FIELDS)
        assert vllm_diagnostics.validate_target_diagnostic(row)  # not full numerical evidence
        runner = setup_state["runner"]
        defs, identity = target_profile.definitions(runner)
        assert identity.bind("internal-0", [1, 2, 3, 7, 8]) in defs
        with pytest.raises((TypeError, ValueError)):
            identity.bind("internal-0", [1, 2, 3, object()])
        with pytest.raises(RuntimeError):
            identity.bind("other-id", [1, 2, 3, 7])
        with pytest.raises(RuntimeError):
            identity.bind("internal-0", [991, 992, 993])
        # Retired bindings retain history. Refill creates a different stable binding.
        other = next(d for d in defs.values() if d.request_id != row["request_id"])
        assert identity.bind("refill", other.prompt_token_ids) == other.request_id
        runner.drafter.identity = FrozenPromptIdentityMap(identity.stable_prompts)
        with pytest.raises(RuntimeError, match="owner changed"):
            target_profile.definitions(runner)
    else:
        assert row["selected_target_token_id"] == [15, 15]
        assert vllm_diagnostics.validate_target_diagnostic(row) == []


@pytest.mark.parametrize("field", ["position_ids", "committed_prefix_sha256",
                                  "target_input_token_ids", "logits_position_mapping"])
def test_lean_still_rejects_missing_required_live_evidence(field, tmp_path, monkeypatch):
    monkeypatch.setenv(target_profile.ENV, "lean")
    row = _capture(tmp_path, monkeypatch, _serial_proposal(), setup=initialize)
    del row[field]
    assert vllm_diagnostics.validate_runtime_target_diagnostic(row)


def test_profile_conflict_and_uninitialized_cache_fail(monkeypatch):
    monkeypatch.setenv(target_profile.ENV, "lean")
    with pytest.raises(RuntimeError, match="not initialized"):
        target_profile.definitions(NS())
    monkeypatch.setenv("SR_PHASE4_NUMERICAL_DIAGNOSTIC_PLAN", "/not-optional")
    with pytest.raises(ValueError, match="conflicts"):
        target_profile.profile()


@pytest.mark.parametrize("selected", ["full", "lean"])
def test_real_cli_manifest_profile_propagation(selected, monkeypatch, tmp_path):
    from specrhythm.serving import decode_scan_cli, decode_scan_plan

    seen = []

    def prepare(root, s1, opts, **kwargs):
        value = decode_scan_plan.manifest(
            {}, [str(i) for i in range(360)], "workload-sha", opts, 128,
            k3_configuration=kwargs["k3_configuration"],
            validation_profile=kwargs["validation_profile"])
        seen.append(value)
        return value

    # Substitute only reading the external S1 fixture; real CLI/options/manifest construction.
    monkeypatch.setattr(decode_scan_cli, "prepare", prepare)
    decode_scan_cli.main([
        "prepare", "--root", str(tmp_path / "root"), "--s1", str(tmp_path / "s1"),
        "--batch", "128", "--k3-configuration", B128,
        "--target-diagnostics", selected, "--draft-dispatch", "unified",
        "--observation", "deferred-window", "--validation-profile", "performance-exploration",
    ])
    diag = seen[0]["fixed_diagnostic"]
    assert diag["options"]["target_diagnostics"] == selected
    assert diag["diagnostic_configuration"]["target_diagnostic_profile"] == selected


@pytest.mark.parametrize("selected", [False, 0, "disabled"])
def test_invalid_profile_is_not_a_missing_optional_diagnostic(selected):
    from specrhythm.serving.fixed_plan import settings

    with pytest.raises(DataError, match="Target diagnostics"):
        settings(target_diagnostics=selected)


@pytest.mark.parametrize("selected", ["full", "lean"])
def test_real_stock_sampling_advances_without_any_forensic_fields(
    selected, runner_class, monkeypatch
):
    from specrhythm.serving.prepost_target import install

    monkeypatch.setenv(target_profile.ENV, selected)
    monkeypatch.setenv("SR_S2_MODE", "pingpong-eager-k3")
    runner = make_runner(runner_class)
    install(runner)
    result = runner._bookkeeping_sync(
        NS(scheduled_spec_decode_tokens={"a": [2, 3, 4], "b": [6, 7, 8]},
           num_scheduled_tokens={"a": 4, "b": 4}),
        NS(sampled_token_ids=Array([[2, 3, 4, 99], [6, 10, -1, -1]]), logprobs_tensors=None),
        None, [0]*8, 8,
    )
    assert result[2] == [[2, 3, 4], [6, 10]]
    assert runner.requests["a"].output_token_ids == [1, 2, 3, 4]
    assert runner.requests["b"].output_token_ids == [1, 6, 10]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("selected", ["full", "lean"])
def test_profile_real_runtime_report_qualification_archive_roundtrip(
    mode, selected, driven, tmp_path
):
    h = driven(mode, "performance", full_run=True, configuration=B128,
               validation_profile="performance-exploration", target_diagnostics=selected)
    options = json.loads(h.path.read_text())["fixed_diagnostic"]["options"]
    target_profile.qualify(h.runtime, options)
    result = decode_scan_results.summarize(h.path, h.directory, h.point, probe=False)
    assert result["valid"], result.get("errors")
    assert result["output_equivalence_status"] == "NOT_RUN"
    archive = tmp_path / "target-profiles.tar.gz"
    export(h.directory, archive, first_code=0, modes=MODES)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        restored = json.load(t.extractfile(paths["runtime.json"]))
    target_profile.qualify(restored, options)
    assert restored["diagnostic_configuration"] == h.runtime["diagnostic_configuration"]
    for bad in ("missing", "wrong"):
        changed = copy.deepcopy(restored)
        worker = changed["target_devices"][0]
        if bad == "missing":
            del worker["diagnostic_configuration"]["target_diagnostic_profile"]
        else:
            worker["diagnostic_configuration"]["target_diagnostic_profile"] = (
                "lean" if selected == "full" else "full")
        with pytest.raises(DataError, match="coverage mismatch"):
            target_profile.qualify(changed, options)


def test_optional_absence_is_explicit_and_mandatory_absence_is_missing():
    result = diagnostic_substages(
        {"diagnostic_configuration": {"target_diagnostic_profile": "lean"}},
        {"target-rank-0": {"intervals": []}}, 1, 10)
    assert result["target_optional_topk_argmax"] == dict(status=target_profile.NOT_COLLECTED)
    assert result["target_diagnostic_contract_scan"]["target-rank-0"]["status"] == "MISSING"


def test_full_and_lean_keep_identical_required_capture_content(tmp_path, monkeypatch):
    monkeypatch.setenv(target_profile.ENV, "full")
    full = _capture(tmp_path / "full", monkeypatch, _serial_proposal(), setup=initialize)
    monkeypatch.setenv(target_profile.ENV, "lean")
    lean = _capture(tmp_path / "lean", monkeypatch, _serial_proposal(), setup=initialize)
    for key in (*target_profile.NUMERICAL_FIELDS, "record_sha256"):
        full.pop(key, None)
    for key in (*target_profile.coverage("lean"), "record_sha256"):
        lean.pop(key, None)
    assert full == lean


def test_ready_wait_does_not_assume_gpu_idle_means_legally_executable():
    from specrhythm.serving.k3_dispatch_evidence import ready_wait

    runtime = dict(measurement_start_ns=0, target_steps=[dict(start_ns=20, end_ns=60)],
                   host={"intervals": []})
    result = ready_wait(runtime, {}, dict(claimed_ns=100, eligibility_observed_ns=90), 10)
    assert result["Target_step_busy_ms"] == 40 / 1e6
    assert result["unaccounted_ms"] == pytest.approx(50 / 1e6)
    assert result["observed_eligibility_to_claim_ms"] == 10 / 1e6
    assert result["first_legal_eligibility_ns"] is None


def test_postprocessing_partition_uses_union_and_preserves_missing():
    from specrhythm.serving.execution_evidence import postprocessing_partition

    row = dict(category="target_forward_diagnostics", pid=1, thread_id=2)
    rank = dict(device={"identity": {"global_rank": 0}}, host={"intervals": [
        dict(row, start_ns=5, end_ns=60), dict(row, start_ns=50, end_ns=80)]})
    result = postprocessing_partition([rank], 10, 100)
    assert result["capture_intersection_union_ms"] == 70 / 1e6
    assert result["outside_capture_ms"] == pytest.approx(20 / 1e6)
    assert postprocessing_partition([rank], None, 100)["status"].startswith("MISSING")
    rank["host"]["intervals"].clear()
    assert postprocessing_partition([rank], 10, 100)["status"].startswith("MISSING")
