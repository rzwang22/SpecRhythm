from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_d3_diagnostics import Tensor
from test_phase4_serial import phase4_config as _config_fixture
from test_phase4_vllm_draft import FakeWorker

from specrhythm.phase4 import draft_logits_probe as probe
from specrhythm.phase4.batched_draft_service import write_immutable_report
from specrhythm.phase4.draft_logits_contract import (
    FIXTURE_PATH,
    FIXTURE_SHA256,
    PATH_SCHEMA,
    PATHS,
    classify_paths,
    load_probe_fixture,
    summarize_logits,
)
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

phase4_config = _config_fixture


def test_frozen_fixture_exactly_matches_original_prefix_and_d3_history(tmp_path):
    fixture = load_probe_fixture()
    old = json.loads(Path("tests/fixtures/phase4b3/d3-b8-8709c81.json").read_text())
    assert fixture["history"] == old["fixtures"][0]
    assert fixture["historical_observed"] == old["observed"]
    assert (
        fixture["source_diagnostic_sha256"]
        == "6483c7c957636d13808570ab8c8538ef145f47a2115bc1ac2fec7e19c2ee3ec4"
    )
    assert len(fixture["patch_sha256"]) == 5
    assert [len(r["committed_prefix"]) for r in fixture["rows"]] == [23, 26, 29, 32, 35, 39, 43]
    assert (
        fixture["rows"][-1]["comparison_prefix_sha256"]
        == "6b15e8e5cb12ec01cc344791bbd8649df17fc5a7225e6cb4a6b20233a236d374"
    )
    changed = tmp_path / "fixture.json"
    modified = copy.deepcopy(fixture)
    modified["rows"][-1]["committed_prefix"][0] = 123
    changed.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="fixture SHA256"):
        load_probe_fixture(changed)


def vector(top, size=151936):
    values = [-1.0] * size
    values[16], values[227] = (14.375, 14.1875) if top == 16 else (14.1875, 14.375)
    return values


def test_full_raw_float32_values_are_not_display_rounded_or_softmaxed():
    values = vector(16, 228)
    values[16], values[227] = 1.0000001192092896, 1.0
    report = summarize_logits(values, "torch.bfloat16")
    restored = json.loads(json.dumps(report))
    assert restored["raw_logits_float32"] == values
    assert restored["token_logits"]["16"] == 1.0000001192092896
    assert report["margin_16_minus_227"] == 2**-23
    assert report["top1_token"] == 16
    values[227] = values[16]
    assert summarize_logits(values, "torch.float32")["top10_token_ids"][:2] == [16, 227]


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, 1.1])
def test_nonfinite_or_lossy_float32_values_rejected(bad):
    values = vector(16, 228)
    values[16] = bad
    with pytest.raises(ValueError):
        summarize_logits(values, "torch.float32")


def frame_for(rows, top):
    ids = ["sr-draft:" + r["request_id"] for r in rows]
    return {
        "prepared_logits_domain": {
            "request_ids": ids,
            "logits_indices": list(range(len(ids))),
            "query_start_loc": list(range(len(ids) + 1)),
        },
        "before_materialize": [
            {
                "internal_id": "sr-draft:" + r["request_id"],
                "request_id": r["request_id"],
                "context_sha256": r["comparison_prefix_sha256"],
                "valid_length": len(r["committed_prefix"]),
                "suffix": [11],
            }
            for r in rows
        ],
        "checks_passed": True,
        "errors": [],
    }


def reports_for(pattern):
    fixture = load_probe_fixture()
    focus = fixture["rows"][-1]
    reports = {}
    for index, (path, top) in enumerate(zip(PATHS, pattern)):
        rows = fixture["rows"][-1:] if path in PATHS[:2] else fixture["rows"]
        by_request = {r["request_id"]: r["expected_hf_next_token"] for r in rows}
        by_request[focus["request_id"]] = top
        reports[path] = {
            "schema_version": PATH_SCHEMA,
            "path": path,
            "valid": True,
            "fixture_sha256": FIXTURE_SHA256,
            "prefix_sha256": focus["comparison_prefix_sha256"],
            "prefix_token_ids": focus["comparison_prefix"],
            "request_order": [r["request_id"] for r in rows],
            "batch_size": len(rows),
            "model_dtype": "bfloat16",
            "initial_live_requests": 0,
            "history_rounds_replayed": 2 if path == PATHS[3] else 0,
            "cleanup_complete": True,
            "process_lifetime_id": f"cpu-process-{index}",
            "run_identity": {"cpu_test_double": True},
            "gpu_identity": {"gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc"},
            "logits": summarize_logits(vector(top), "torch.bfloat16"),
            "top1_by_request": by_request,
            "runtime_provenance": {"batch_invariance": {"batch_invariant_env_resolved": True}},
            "captured_frame": frame_for(rows, top),
        }
    return reports


@pytest.mark.parametrize(
    "pattern,case",
    [
        ([16, 227, 227, 227], "A"),
        ([16, 16, 16, 227], "B"),
        ([16, 16, 227, 227], "C"),
        ([16, 16, 16, 16], None),
        ([227, 227, 227, 227], None),
        ([16, 227, 16, 227], None),
    ],
)
def test_four_path_classification_is_evidence_not_automatic_fix(pattern, case):
    result = classify_paths(load_probe_fixture(), reports_for(pattern))
    assert not result["errors"]
    assert result["classification_case"] == case
    assert result["evidence_status"] == ("strong_evidence" if case else "inconclusive")
    assert result["top1_pattern"] == pattern
    assert not result["mechanistic_root_cause_proven"]
    assert not result["correctness_fix_authorized_by_result"] and not result["d4_d5_allowed"]
    assert result["persistent_failure_reproduced"] is (pattern[-1] == 227)
    if case == "B":
        assert any("kv_content" in s for s in result["required_followups"])
    if case == "C":
        assert any("batch_invariance" in s for s in result["required_followups"])


def test_equal_top1_but_different_raw_b_c_requires_source_audit():
    reports = reports_for([16, 227, 227, 227])
    values = reports[PATHS[2]]["logits"]["raw_logits_float32"]
    values[16] += 0.0625
    reports[PATHS[2]]["logits"] = summarize_logits(values, "torch.bfloat16")
    result = classify_paths(load_probe_fixture(), reports)
    assert result["classification_case"] == "A"
    assert any("batch_invariance" in s for s in result["required_followups"])
    assert (
        result["raw_logit_deltas"]["vllm-fresh-b7_minus_vllm-fresh-singleton"]["token_logits"][
            "16"
        ]
        == 0.0625
    )


@pytest.mark.parametrize(
    "fault",
    [
        "prefix",
        "order",
        "raw",
        "history",
        "state",
        "cleanup",
        "domain",
        "lifetime",
        "identity",
        "missing",
    ],
)
def test_invalid_or_contaminated_path_never_receives_positive_classification(fault):
    reports = reports_for([16, 16, 16, 227])
    row = reports[PATHS[2]]
    if fault == "prefix":
        row["prefix_sha256"] = "0" * 64
    elif fault == "order":
        row["request_order"] = list(reversed(row["request_order"]))
    elif fault == "raw":
        row["logits"]["token_logits"]["16"] = 999.0
    elif fault == "history":
        row["history_rounds_replayed"] = 1
    elif fault == "state":
        row["initial_live_requests"] = 1
    elif fault == "cleanup":
        row["cleanup_complete"] = False
    elif fault == "domain":
        row["captured_frame"]["prepared_logits_domain"]["logits_indices"][-1] = 0
    elif fault == "lifetime":
        row["process_lifetime_id"] = reports[PATHS[1]]["process_lifetime_id"]
    elif fault == "identity":
        row["run_identity"] = {"different_weights": True}
    elif fault == "missing":
        del reports[PATHS[3]]
    result = classify_paths(load_probe_fixture(), reports)
    assert result["errors"] and result["classification_case"] is None
    assert result["evidence_status"] == "invalid"


class PrivateWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.views = self.memory


@pytest.mark.parametrize("path", PATHS[1:3])
def test_fresh_controls_have_only_scratch_prefills_then_exact_b1_or_b7_step(phase4_config, path):
    fixture = load_probe_fixture()
    worker = PrivateWorker()
    backend = VllmBatchedDraftBackend(phase4_config, worker=worker)
    rows = probe.initialize_fresh(backend, fixture, path)
    probe.run_fresh_forward(backend, rows)
    assert len(worker.calls) == len(rows) + 1
    assert all(
        purpose == "setup" and len(plans) == 1 and plans[0].valid_length == 0
        for purpose, plans in worker.calls[:-1]
    )
    purpose, plans = worker.calls[-1]
    assert purpose == "proposal" and len(plans) == (1 if path == PATHS[1] else 7)
    assert [p.request_id for p in plans] == ["sr-draft:" + r["request_id"] for r in rows]
    for row, plan in zip(rows, plans):
        assert list(plan.context) == row["comparison_prefix"] and plan.suffix == (11,)
        assert plan.valid_length == len(row["committed_prefix"])
        assert worker.memory[plan.request_id] == row["comparison_prefix"]
    before = len(worker.calls)
    with pytest.raises(ValueError, match="inherited"):
        probe.initialize_fresh(backend, fixture, path)
    assert len(worker.calls) == before
    backend.shutdown()


def test_persistent_control_replays_original_b8_eos_b7_corrections(phase4_config):
    fixture = load_probe_fixture()
    history = fixture["history"]
    prefixes = {r: tuple(p) for r, p in history["initial"].items()}
    next_tokens = {}
    for item in history["rounds"]:
        for rid, tokens in item["expected"].items():
            for step, token in enumerate(tokens):
                next_tokens[prefixes[rid] + tuple(tokens[:step])] = token
        for row in item["synchronizations"]:
            prefixes[row["request_id"]] += tuple(row["committed_delta"])

    class Oracle(PrivateWorker):
        def materialize(self, rows, purpose):
            super().materialize(rows, purpose)
            return {
                r.request_id: next_tokens.get(
                    tuple(self.memory[r.request_id][: len(r.context)]), 0
                )
                for r in rows
            }

    worker = Oracle()
    backend = VllmBatchedDraftBackend(phase4_config, worker=worker)
    observer = SimpleNamespace(attach=lambda: None, round=None)
    comparisons = probe.replay_persistent(backend, fixture, observer)
    assert [r["exact"] for r in comparisons] == [True, True, True]  # CPU oracle; no GPU claim.
    assert observer.round == 2
    measured = [(purpose, plans) for purpose, plans in worker.calls if purpose != "setup"]
    assert len(measured) == 11 and len(measured[0][1]) == 8
    assert len(measured[4][1]) == 7
    assert "sr-draft:batch-8-0" not in worker.memory
    for plan, frozen in zip(measured[8][1], fixture["rows"]):
        assert token_prefix_hash(plan.context) == frozen["comparison_prefix_sha256"]
        assert plan.valid_length == len(frozen["committed_prefix"])
    backend.shutdown()


def test_raw_capture_uses_actual_logits_row_and_leaves_tensor_unchanged():
    fixture = load_probe_fixture()
    row = fixture["rows"][-1]
    rid = "sr-draft:" + row["request_id"]

    class TypedTensor(Tensor):
        dtype = "torch.bfloat16"

        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(result, Tensor):
                result.dtype = self.dtype
            return result

    logits = TypedTensor([vector(227, 228)])
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=[rid]))
    backend = SimpleNamespace(
        worker=SimpleNamespace(runner=runner),
        states={row["request_id"]: SimpleNamespace(internal_id=rid)},
    )
    observer = probe.ProbeCapture(backend, fixture, PATHS[1])
    observer.frame = {**frame_for([row], 227), "index": 0, "round": 2, "purpose": "proposal"}
    wrapped = observer._wrap_logits(lambda hidden: logits)
    assert wrapped(None) is logits
    assert observer.capture["logits"]["token_logits"] == {"16": 14.1875, "227": 14.375}
    assert observer.capture["top1_by_request"] == {row["request_id"]: 227}
    assert observer.capture["logits"]["raw_logits_float32"] == logits.data[0]


@pytest.mark.parametrize("failed_path", [None, PATHS[2]])
def test_orchestrator_runs_four_distinct_sequential_processes_and_keeps_failures(
    monkeypatch, tmp_path, failed_path
):
    fixture = load_probe_fixture()
    reports = reports_for([16, 16, 227, 227])
    identity = reports[PATHS[0]]["run_identity"]
    monkeypatch.setattr(probe, "preflight", lambda *a: (None, identity))
    events, active = [], []

    class Process:
        def __init__(self, command, stdout, stderr):
            assert not active
            self.path = command[command.index("--path") + 1]
            self.root = Path(command[command.index("--output") + 1])
            assert "--allow-gpu" in command
            assert json.loads((self.root / "probe-inputs.json").read_text())["fixture"] == fixture
            self.pid = 100 + len(events)
            active.append(self)
            events.append(self.path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            active.remove(self)

        def wait(self):
            if self.path == failed_path:
                return 9
            write_immutable_report(
                self.root / f"{self.path}.json", {**reports[self.path], "process_pid": self.pid}
            )
            return 0

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    output = tmp_path / "probe"
    result = probe.run_all(Path("config.json"), "0" * 40, output)
    assert events == list(PATHS) and not active
    assert (output / "classification.json").is_file()
    assert all((output / f"{p}.json").is_file() for p in PATHS)
    assert result["evidence_status"] == ("invalid" if failed_path else "strong_evidence")
    with pytest.raises(FileExistsError):
        probe.run_all(Path("config.json"), "0" * 40, output)
    assert events == list(PATHS)


def test_probe_import_and_missing_allow_gpu_do_not_load_gpu_packages():
    code = (
        "import sys; import specrhythm.phase4.draft_logits_probe; "
        "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "specrhythm.phase4.draft_logits_probe",
            "--config",
            "missing",
            "--expected-commit",
            "0" * 40,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "--allow-gpu" in result.stderr
    assert "No such file" not in result.stderr


def test_frozen_fixture_is_packaged_and_serving_paths_do_not_import_probe():
    assert FIXTURE_PATH.name in Path("pyproject.toml").read_text()
    for name in (
        "draft_service.py",
        "dual_service.py",
        "vllm_draft_backend.py",
        "vllm_draft_worker.py",
    ):
        assert "draft_logits_probe" not in Path("src/specrhythm/phase4", name).read_text()
