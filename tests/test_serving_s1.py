"""Synthetic CPU contracts only: no GPU numerical, KV, overlap or throughput qualification."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_phase4_dual_scheduler import scheduler as _scheduler
from test_phase4_pingpong import pingpong_factory as _pingpong_factory
from test_phase4_resident_setup import _provenance
from test_phase4_serial import phase4_config as _phase4_config

from specrhythm.phase4.decode_ready import ResidentSetupObservation, ResidentWarmStartProvider
from specrhythm.phase4.dual_rhythm import balanced_assignment
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.performance_boundary import (
    PERFORMANCE_COMMIT_SCHEMA,
    PERFORMANCE_EVENT,
    PERFORMANCE_EVENT_SCHEMA,
)
from specrhythm.phase4.process_lifecycle import LIFECYCLE_SCHEMA, run_owned_target
from specrhythm.phase4.resident_setup import (
    build_deferred_initial_proposals_ready,
    load_deferred_initial_proposals_ready,
)
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.stock_vllm import load_smoke_requests
from specrhythm.phase4.vllm_diagnostics import DIAGNOSTIC_SCHEMA
from specrhythm.serving.common import REQUEST_SCHEMA, TASKS, digest, read_json, text_hash
from specrhythm.serving.s1_cli import previous_completed, run_plan, start
from specrhythm.serving.s1_policy import (
    EFFECTIVE_SCHEMA,
    G0_SCHEMA,
    REQUIRED_ENVIRONMENT,
    RESULT_SCHEMA,
    policy_fields,
)
from specrhythm.serving.s1_preflight import capacity_estimate, clean_environment
from specrhythm.serving.s1_results import (
    compare_results,
    draft_measurement,
    inspect_run,
    interval_duration,
    seal_run,
    validate_outputs,
    validate_rounds,
    verify_seal,
    workload_metrics,
)
from specrhythm.serving.s1_runtime import resident_capacity, run_consumer
from specrhythm.serving.s1_workload import (
    PROFILE_ENV,
    QUOTAS,
    SCHEMA,
    create_manifest,
    initial_proposal_excluded_ids,
    load_execution,
    load_runtime_requests,
    select_subset,
    setup_terminal_ids,
    target_options,
    validate_prompt_ids,
    write_once,
)
from specrhythm.serving.schema import ServingWorkloadRequest

phase4_config = _phase4_config
pingpong_factory = _pingpong_factory
scheduler = _scheduler


def dump_rows(path, rows):
    from specrhythm.phase4.transport import CheckpointJsonl

    path.write_text("")
    for row in rows:
        CheckpointJsonl(path).append({k: v for k, v in row.items() if k != "record_sha256"})


def request(index=0, budget=3):
    text = f"<synthetic-prompt-{index}>"
    return ServingWorkloadRequest(
        schema_version=REQUEST_SCHEMA,
        workload_id="synthetic-s1",
        split="main",
        request_id=f"r{index}",
        task_class=TASKS[index % 4],
        language="English",
        source_ref=dict(
            dataset_id="synthetic",
            revision="a" * 40,
            config="test",
            split="train",
            id=str(index),
            file="synthetic.jsonl",
            file_sha256="b" * 64,
            original_row_index=index,
            group_id=str(index),
        ),
        messages=[dict(role="user", content=text)],
        prompt_text=text,
        prompt_token_ids=[10 + index * 2, 11 + index * 2],
        prompt_length=2,
        prompt_sha256=text_hash(text),
        tokenizer_fingerprint="c" * 64,
        arrival_time_ms=float(index),
        arrival_source_ref=dict(
            repository_id="synthetic",
            file="arrivals",
            revision="a" * 40,
            file_sha256="b" * 64,
            original_line_index=index + 1,
            source_timestamp_ms=float(index),
        ),
        maximum_new_tokens=budget,
        sampling_seed=index,
        slo_class=TASKS[index % 4],
    )


def execution(tmp_path, rows=None):
    rows = rows or [request(i) for i in range(4)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    file = tmp_path / "requests.jsonl"
    file.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    ids = [r.request_id for r in rows]
    logical = {
        "request_ids": ids,
        "workload_sha256": sha256_file(file),
        "arrival_replay_enabled": False,
        "requests": [
            dict(
                request_id=r.request_id,
                maximum_new_tokens=r.maximum_new_tokens,
                sampling_seed=r.sampling_seed,
            )
            for r in rows
        ],
    }
    value = {
        "schema_version": SCHEMA,
        **policy_fields(),
        "logical": logical,
        "logical_sha256": digest(logical),
        "request_ids_sha256": digest(ids),
        "workload_file": file.name,
        "assignment": {rid: "AB"[i % 2] for i, rid in enumerate(ids)},
        "parent": {},
        "execution": {
            "git_commit": "1" * 40,
            "eos_token_ids": [999],
            "capacity": {
                "max_model_len": 4096,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 4096,
            },
        },
    }
    value["manifest_sha256"] = digest(value)
    path = tmp_path / "execution-manifest.json"
    write_once(path, value)
    return path, value, rows


def freeze_serial_inputs(root, path, manifest):
    """Freeze real config/patch-report shapes, never a precreated decode-ready context."""
    from test_phase4 import CONFIG

    from specrhythm.phase4.serial_runner import (
        PATCHED_VLLM_RUNNER_SHA256,
        PATCHED_VLLM_SCHEDULER_SHA256,
    )
    from specrhythm.serving.s1_preflight import REPO

    write_once(root / "config.json", json.loads(CONFIG.read_text()))
    patches = REPO / "integrations/vllm/patches"
    names = (
        "0001-custom-proposer-request-and-verify-hooks.patch",
        "0002-scheduler-request-admissibility-hook.patch",
        "0003-target-forward-timing-observer.patch",
        "0004-gate3-numerical-observer.patch",
        "0005-dual-sampled-row-context.patch",
    )
    write_once(root / "patch-manifest.json", dict(
        schema_version="specrhythm.vllm-patch-state-check.v1",
        operation="check", expected_state="patched", valid=True, errors=[],
        verified_source_commit="752a3a504485790a2e8491cacbb35c137339ad34",
        actual_runner_sha256=PATCHED_VLLM_RUNNER_SHA256,
        actual_scheduler_sha256=PATCHED_VLLM_SCHEDULER_SHA256,
        patch_stack_applied=True,
        patch_stack=[dict(patch_file=n, patch_sha256=sha256_file(patches / n)) for n in names],
    ))
    manifest["execution"].update(
        config_sha256=sha256_file(root / "config.json"),
        patch_manifest_sha256=sha256_file(root / "patch-manifest.json"),
        numerical_mode="batch-invariant", K=4, Draft_backend="vllm-batched", uuid_mode="live",
    )
    manifest["manifest_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    path.write_text(json.dumps(manifest))


def ready_manifest(rows, execution_manifest, bootstraps=None):
    bootstraps = bootstraps or [100 + i for i in range(len(rows))]
    observations = [
        ResidentSetupObservation(
            request_id=r.request_id,
            internal_target_request_id="opaque-" + r.request_id,
            prompt_token_ids=tuple(r.prompt_token_ids),
            bootstrap_token_id=b,
            target_materialized_kv_token_count=r.prompt_length,
            target_num_computed_tokens=r.prompt_length,
            draft_materialized_kv_token_count=r.prompt_length + 1,
            bootstrap_ready_ns=20,
            draft_initialization_complete_ns=25,
        )
        for r, b in zip(rows, bootstraps)
    ]
    provenance = replace(
        _provenance(), workload_sha256=execution_manifest["logical"]["workload_sha256"]
    )
    return ResidentWarmStartProvider().prepare(
        observations,
        provenance,
        setup_start_ns=10,
        setup_complete_ns=40,
        global_barrier_ns=50,
        measurement_start_ns=60,
    )


def outputs(rows, tokens):
    return [
        dict(
            request_id=r.request_id,
            prompt_length=r.prompt_length,
            generated_token_ids=t,
            generated_tokens=len(t),
            finish_reason="stop" if t[-1] == 999 else "length",
            stop_reason=None,
            token_accounting=dict(
                prompt_tokens=r.prompt_length,
                generated_tokens=len(t),
                total_tokens=r.prompt_length + len(t),
            ),
        )
        for r, t in zip(rows, tokens)
    ]


@pytest.mark.parametrize("count", [1, 3, 4, 7, 20, 100])
def test_four_class_arbitrary_runtime_profile_and_legacy_unchanged(tmp_path, monkeypatch, count):
    path, _, source = execution(tmp_path, [request(i) for i in range(count)])
    monkeypatch.setenv(PROFILE_ENV, str(path))
    actual = load_runtime_requests(tmp_path / "requests.jsonl", count)
    assert [r.source.to_dict() for r in actual] == [r.to_dict() for r in source]
    assert all(type(r.prompt_token_ids) is tuple for r in actual)
    assert dict(balanced_assignment(r.request_id for r in actual)) == read_json(path)["assignment"]
    with pytest.raises((ValueError, KeyError)):
        load_smoke_requests(tmp_path / "requests.jsonl", count, require_task_mixture=False)
    monkeypatch.delenv(PROFILE_ENV)
    assert target_options() == {}
    with pytest.raises((ValueError, KeyError)):
        load_runtime_requests(tmp_path / "requests.jsonl", count, False)


def test_subsets_deterministic_nested_original_order_and_parent_binding(tmp_path):
    rows = [request(i) for i in range(160)]
    sets = []
    for subset, quota in QUOTAS.items():
        chosen = select_subset(rows, quota)
        ids = [r.request_id for r in chosen]
        assert ids == [r.request_id for r in rows if r.request_id in ids]
        sets.append(set(ids))
        a = create_manifest(
            tmp_path / subset / "a",
            subset,
            rows,
            {"test": "synthetic"},
            {"capacity": {"max_model_len": 4096, "max_num_seqs": 128}},
        )
        b = create_manifest(
            tmp_path / subset / "b",
            subset,
            rows,
            {"test": "synthetic"},
            {"capacity": {"max_model_len": 4096, "max_num_seqs": 128}},
        )
        assert a == b
        assert load_execution(tmp_path / subset / "a/execution-manifest.json")[0] == a
    assert sets[0] < sets[1] < sets[2]


@pytest.mark.parametrize("mutation", ["tokens", "manifest", "count", "assignment"])
def test_profile_refuses_wrong_hash_count_or_assignment(tmp_path, monkeypatch, mutation):
    path, value, _ = execution(tmp_path)
    if mutation == "tokens":
        (tmp_path / "requests.jsonl").write_text("{}\n")
    elif mutation == "manifest":
        value["execution"]["capacity"]["max_num_seqs"] = 1
        path.write_text(json.dumps(value))
    elif mutation == "assignment":
        value["assignment"]["r0"] = "B"
        value["manifest_sha256"] = digest(
            {k: v for k, v in value.items() if k != "manifest_sha256"}
        )
        path.write_text(json.dumps(value))
    monkeypatch.setenv(PROFILE_ENV, str(path))
    with pytest.raises(ValueError):
        load_runtime_requests(tmp_path / "requests.jsonl", 99 if mutation == "count" else 4)


def test_no_template_reapplication_or_special_token_insertion(tmp_path, monkeypatch):
    path, _, rows = execution(tmp_path)
    monkeypatch.setenv(PROFILE_ENV, str(path))
    calls = []
    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: calls.append((text, kw)) or rows[0].prompt_token_ids
    )
    validate_prompt_ids(tokenizer, rows[0])
    assert calls == [(rows[0].prompt_text, {"add_special_tokens": False})]


def test_natural_eos_short_budget_zero_timed_tokens(tmp_path, monkeypatch):
    rows = [request(0, 512), request(1, 1), request(2, 2), request(3, 3)]
    path, manifest, _ = execution(tmp_path, rows)
    monkeypatch.setenv(PROFILE_ENV, str(path))
    ready = ready_manifest(rows, manifest, [999, 101, 102, 103])
    tokens = [[999], [101], [102, 999], [103, 104, 105]]
    events = [
        dict(request_id="r2", token_ids=[999], timestamp_ns=200),
        dict(request_id="r3", token_ids=[104, 105], timestamp_ns=210),
    ]
    result = validate_outputs(rows, outputs(rows, tokens), [999], ready, events, 100, 300)
    assert [r["timed_committed_tokens"] for r in result] == [0, 0, 1, 2]
    assert setup_terminal_ids(ready) == {"r0", "r1"}
    assert initial_proposal_excluded_ids(ready) == {"r0", "r1", "r2"}
    metrics = workload_metrics(result, 100, 300)
    assert metrics["timed_committed_tokens"] == 3
    assert metrics["decode_makespan_ms"] == 0.0002
    assert "not serving TPOT" in metrics["time_scope"]


@pytest.mark.parametrize("fault", ["after_eos", "termination", "count", "missing", "fixed_work"])
def test_actual_output_accounting_failure_blocks(tmp_path, fault):
    rows = [request()]
    _, manifest, _ = execution(tmp_path, rows)
    ready = ready_manifest(rows, manifest)
    out = outputs(rows, [[100, 101, 999]])
    events = [dict(request_id="r0", token_ids=[101, 999], timestamp_ns=200)]
    if fault == "after_eos":
        out[0]["generated_token_ids"] = [100, 999, 102]
    if fault == "termination":
        out[0]["finish_reason"] = "length"
    if fault == "count":
        out[0]["generated_tokens"] = 1487
    if fault == "missing":
        out = []
    if fault == "fixed_work":
        events[0]["token_ids"] = [101] * 16
    with pytest.raises(ValueError):
        validate_outputs(rows, out, [999], ready, events, 100, 300)


def test_empty_deferred_proposals_have_authoritative_terminal_evidence(tmp_path, monkeypatch):
    path, manifest, rows = execution(tmp_path, [request(0, 1)])
    monkeypatch.setenv(PROFILE_ENV, str(path))
    ready = ready_manifest(rows, manifest)
    ready_path = tmp_path / "decode-ready.json"
    write_once(ready_path, ready.to_dict())
    value = build_deferred_initial_proposals_ready(
        ready, proposals=[], performance_measurement_start_ns=100, published_ns=110
    )
    write_once(tmp_path / "deferred.json", value)
    assert (
        load_deferred_initial_proposals_ready(
            tmp_path / "deferred.json", manifest_path=ready_path, expected_request_ids=["r0"]
        )
        == ()
    )
    monkeypatch.delenv(PROFILE_ENV)
    with pytest.raises(ValueError, match="request set"):
        build_deferred_initial_proposals_ready(
            ready, proposals=[], performance_measurement_start_ns=100, published_ns=110
        )


def test_real_serial_initial_generation_excludes_terminal_and_target_tail(tmp_path, monkeypatch):
    from specrhythm.phase4.vllm_remote import RemoteDraftProposer

    path, _, rows = execution(tmp_path, [request(0, 1), request(1, 2)])
    monkeypatch.setenv(PROFILE_ENV, str(path))
    proposer = object.__new__(RemoteDraftProposer)
    proposer.definitions = {r.request_id: r for r in rows}
    proposer.requests = {
        r.request_id: SimpleNamespace(
            generated_token_ids=(100,), finished=r.maximum_new_tokens == 1
        )
        for r in rows
    }
    proposer.client = SimpleNamespace(
        call=lambda *a: pytest.fail("terminal/tail must not invoke Draft")
    )
    assert proposer._generate_initial_resident_proposals(100) == ()


@pytest.mark.parametrize("mode", ["target", "serial", "pingpong", "raw-target"])
def test_actual_consumer_entry_receives_s1_manifest_and_full_budgets(
    tmp_path, monkeypatch, phase4_config, mode
):
    from specrhythm.phase4 import dual_runner, resident_runner, serial_runner, stock_vllm
    from specrhythm.serving import s1_runtime

    for key, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    for key in ("USE_TORCH", "USE_TF", "USE_FLAX"):
        monkeypatch.delenv(key, raising=False)

    path, manifest, rows = execution(
        tmp_path / "profile", [request(i, 512 if i % 2 == 0 else 1024) for i in range(4)]
    )
    directory = tmp_path / "run"
    directory.mkdir()
    if mode == "serial":
        freeze_serial_inputs(tmp_path, path, manifest)
    monkeypatch.setattr(s1_runtime, "load_execution", lambda *a, **k: (manifest, rows))
    monkeypatch.setattr(s1_runtime, "load_phase4_config", lambda *a: phase4_config)
    monkeypatch.setenv("SR_S1_DRAFT_SOCKET", "/tmp/synthetic-only.sock")
    module, name = {
        "target": (resident_runner, "run_resident_target"),
        "serial": (serial_runner, "run_serial_disaggregated"),
        "pingpong": (dual_runner, "run_resident_dual_batch"),
        "raw-target": (stock_vllm, "run_stock_smoke"),
    }[mode]
    signature = inspect.signature(getattr(module, name))
    captured = []

    def fake(*args, **kwargs):
        signature.bind(*args, **kwargs)  # Catch wrong real-runner keyword names.
        actual = load_runtime_requests(kwargs["workload_path"], kwargs["request_count"])
        assert [r.maximum_new_tokens for r in actual] == [512, 1024, 512, 1024]
        captured.append(kwargs)
        return {
            "valid": True,
            "repeated_run_deterministic": False,
            "runs": [outputs(rows, [[100 + i, 999] for i in range(4)])],
            "synthetic_CPU_contract": True,
        }

    monkeypatch.setattr(module, name, fake)
    if mode == "raw-target":
        monkeypatch.setattr(serial_runner, "load_patch_manifest", lambda *a: {})
        monkeypatch.setattr(serial_runner, "validate_installed_patch_stack", lambda *a: {})
    assert run_consumer(mode, path, directory, tmp_path)["valid"]
    assert len(captured) == 1
    if mode != "raw-target":
        assert captured[0]["phase4b2_performance"] is True
    if mode in ("target", "serial"):
        assert captured[0]["reference_path"] is None
    if mode == "raw-target":
        assert captured[0]["compare_repeated_outputs"] is False


def test_round_terminal_truncation_no_fixed_plus_one():
    request_result = {"request_id": "r0", "generated_token_ids": [100, 5, 999]}
    row = dict(
        request_id="r0",
        round_id=0,
        parent_prefix_len=3,
        parent_prefix_hash="a" * 64,
        proposal_token_ids=[5, 999, 8],
        accepted_draft_token_ids=[5, 999],
        rejected_draft_token_ids=[8],
        accepted_draft_tokens=2,
        rejected_draft_tokens=1,
        target_correction_token_ids=[],
        target_bonus_token_ids=[],
        committed_token_ids=[5, 999],
    )
    assert validate_rounds([row], [request_result], {"r0": 2}, [999])[0]["terminal"]
    with pytest.raises(ValueError):
        validate_rounds([row, row], [request_result], {"r0": 2}, [999])
    row["accepted_draft_tokens"] = 3
    with pytest.raises(ValueError):
        validate_rounds([row], [request_result], {"r0": 2}, [999])


def test_tp_parallel_intervals_union_not_sum_and_empty_optional_stats():
    assert interval_duration([(10, 30), (10, 30), (20, 40)]) == 30
    assert interval_duration([]) == 0
    from specrhythm.serving.s1_results import distribution

    assert distribution([]) == dict(count=0, p50=None, p90=None, p99=None, min=None, max=None)


def synthetic_result(mode="target"):
    row = dict(
        request_id="r0",
        task_class="chat",
        generated_token_ids=[1, 2],
        generated_tokens=2,
        finish_reason="length",
        stop_reason=None,
        bootstrap_token_ids=[1],
        final_prefix_sha256="a" * 64,
        timed_committed_tokens=1,
        untimed_output_tokens=1,
        terminal_in_setup=False,
        decode_barrier_to_completion_ms=1,
        eos_terminated=False,
        max_token_termination=True,
    )
    return dict(
        schema_version=RESULT_SCHEMA,
        **policy_fields(),
        mode=mode,
        valid=True,
        performance_result=True,
        execution_sha256="a" * 64,
        checks={k: dict(valid=True) for k in ("correctness", "measurement", "lifecycle")},
        requests=[row],
        round_semantics=[],
        metrics=workload_metrics([row], 100, 1000100),
    )


@pytest.mark.parametrize(
    "fault", [None, "lifecycle", "missing", "identity", "work"]
)
def test_three_mode_comparator_retains_validity_and_internal_metric_accounting(fault):
    results = {mode: [synthetic_result(mode)] for mode in ("target", "serial", "pingpong")}
    bad = results["pingpong"][0]
    if fault == "lifecycle":
        bad["checks"]["lifecycle"]["valid"] = False
    if fault == "missing":
        bad["requests"] = []
    if fault == "identity":
        bad["execution_sha256"] = "b" * 64
    if fault == "work":
        bad["requests"][0]["timed_committed_tokens"] = 1487
    report = compare_results(results)
    assert report["valid"] is (fault is None)
    assert (report["performance_ratios"] is not None) is (fault is None)
    assert report["repeatability"]["exact_tokens_and_termination"] is None
    assert report["matched_work"]["exact_timed_output_equal"] is None


def test_rotation_and_environment_disable_inference_flags_cleared(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "0")
    monkeypatch.setenv("SR_PHASE4_DUAL_UUID_QUERY_MODE", "cached")
    monkeypatch.setenv("SR_PHASE4_NUMERICAL_PLAN", "stale")
    env = clean_environment("pingpong")
    assert "USE_TORCH" not in env and "SR_PHASE4_NUMERICAL_PLAN" not in env
    assert env["SR_PHASE4_DUAL_UUID_QUERY_MODE"] == "live"
    assert all(env[k] == v for k, v in REQUIRED_ENVIRONMENT.items())
    assert run_plan("G3") == [
        ("target", 0),
        ("serial", 0),
        ("pingpong", 0),
        ("serial", 1),
        ("pingpong", 1),
        ("target", 1),
        ("pingpong", 2),
        ("target", 2),
        ("serial", 2),
    ]
    assert run_plan("G1").count(("pingpong", 1)) == 1
    assert run_plan("G1") == [("target", 0), ("serial", 0), ("pingpong", 0), ("pingpong", 1)]


def test_capacity_rejects_without_reducing_work():
    rows = [request(i, 1024) for i in range(100)]
    cfg = dict(
        hidden_size=8192, num_attention_heads=64, num_key_value_heads=8, num_hidden_layers=64
    )
    report = capacity_estimate(rows, cfg, 64 * 1024**3, 2, 1024, 0.85)
    assert report["valid"] is False and report["request_count"] == 100
    assert resident_capacity(rows, 16)["maximum_logical_tokens"] == 103000


def test_seal_resume_detects_mutation_and_never_reuses_incomplete(tmp_path):
    run = tmp_path / "attempt-001"
    run.mkdir()
    write_once(run / "raw.json", {"synthetic": True})
    assert previous_completed([run], "a" * 64) is None
    report = synthetic_result()
    seal_run(run, report)
    assert previous_completed([run], "a" * 64)[0] == run
    (run / "raw.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_seal(run)


def test_launcher_detaches_and_records_real_exit_status(tmp_path, monkeypatch):
    _, manifest, _ = execution(tmp_path / "s1-smoke4")
    write_once(tmp_path / "g0.json", {
        "schema_version": G0_SCHEMA, **policy_fields(),
        "valid": True, "execution": manifest["execution"],
    })
    launched = []

    def fake(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(subprocess, "Popen", fake)
    assert start(tmp_path, "G1") == 0
    assert launched[0][1]["start_new_session"] is True
    assert launched[0][1]["stdin"] is subprocess.DEVNULL
    assert "supervise" in launched[0][0]


def test_owned_cpu_command_timeout_and_failure_exit_are_preserved(tmp_path):
    for name, code, timeout, expected in (
        ("fail", "raise SystemExit(7)", None, 7),
        ("timeout", "import time; time.sleep(30)", 0.2, 124),
    ):
        status, report = run_owned_target(
            [sys.executable, "-c", code],
            target_log=tmp_path / (name + ".log"),
            artifact_path=tmp_path / (name + ".json"),
            timeout_seconds=timeout,
            ownership_journal=tmp_path / (name + "-ownership.json"),
            graceful_seconds=0.2,
            kill_seconds=0.2,
            poll_seconds=0.02,
        )
        assert status == expected
        assert report["owned_cleanup_completed"] and not report["remaining_owned_pids"]
        assert (
            report["target_exit_status"] == 7
            if name == "fail"
            else report["target_exit_status"] != 0
        )


def test_cli_cpu_help_does_not_import_inference():
    result = subprocess.run(
        [sys.executable, "-m", "specrhythm.serving.s1_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "prepare" in result.stdout
    assert "torch" not in sys.modules and "vllm" not in sys.modules


def native_artifacts(
    tmp_path, monkeypatch, mode="target", zero=False, speculative=False,
    *, token_offset=0, setup_terminal=(), eos_finish=(),
):
    """Synthetic native-schema artifacts: validators run normally; only S0 parent is fake."""
    from test_phase4_vllm_draft import FakeWorker

    from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
    from specrhythm.serving import s1_results

    path, manifest, definitions = execution(
        tmp_path / "input",
        [request(i, 512 if zero else 3 if speculative else 2) for i in range(4)],
    )
    monkeypatch.setattr(s1_results, "load_execution", lambda path, **kw: load_execution(path))
    monkeypatch.setenv(PROFILE_ENV, str(path))
    run = tmp_path / "run"
    run.mkdir()
    terminal = set(range(4)) if zero else set(setup_terminal)
    bootstraps = [999 if i in terminal else 100 + token_offset + i for i in range(4)]
    ready = ready_manifest(definitions, manifest, bootstraps)
    final_tokens = [
        [b] if i in terminal else [b, 999 if i in eos_finish else 200 + token_offset + i]
        for i, b in enumerate(bootstraps)
    ]
    if speculative:
        final_tokens = [tokens + [300 + i] for i, tokens in enumerate(final_tokens)]
    raw = dict(
        valid=True,
        errors=None,
        outputs=outputs(definitions, final_tokens),
        phase4b2_final_sync=[
            dict(local_rank=i, physical_gpu_id=i + 1, final_cuda_synchronize_complete_ns=500 + i)
            for i in range(2)
        ],
    )
    if mode == "pingpong":
        raw.update(dual_rhythm="pingpong", cohort_assignment=manifest["assignment"])
        raw["decode_only_outputs"] = [
            {**row, "final_logical_length": row["prompt_length"] + row["generated_tokens"]}
            for row in raw["outputs"]
        ]
    write_once(run / "raw.json", raw)
    write_once(run / "decode-ready-manifest.json", ready.to_dict())
    write_once(run / "setup-ready.json", dict(global_decode_ready=True, ready_published_ns=90))
    write_once(
        run / "s1-effective-runtime.json",
        dict(
            schema_version=EFFECTIVE_SCHEMA, **policy_fields(),
            execution_sha256=manifest["manifest_sha256"],
        ),
    )
    write_once(run / "exit-code.json", dict(effective_exit_code=0, coordinator_exit_code=0))
    write_once(
        run / "process-lifecycle.json",
        dict(
            schema_version=LIFECYCLE_SCHEMA,
            coordinator_pid=111,
            pgid=111,
            session_id=111,
            child_reap_result=dict(coordinator_reaped=True),
            target_exit_status=0,
            remaining_owned_pids=[],
            cleanup_valid=True,
            run_valid=True,
        ),
    )
    # Exercise the actual report producer, not a hand-written backend selector/name.
    draft_backend = VllmBatchedDraftBackend(
        SimpleNamespace(max_model_len=4096), worker=FakeWorker()
    )
    draft_metrics = draft_backend.metrics
    if speculative:
        draft_metrics.forward("proposal", 4, 4)
        draft_metrics.gpu_ms["proposal"] = 0.1
        draft_metrics.s1_forward_records.append(
            dict(
                purpose="proposal",
                B=4,
                Q=4,
                host_start_ns=110,
                host_launch_end_ns=150,
                cuda_completion_observed_ns=900,
                gpu_event_ms=0.1,
            )
        )
        dump_rows(
            run / "round-events.jsonl",
            [
                dict(
                    request_id=r.request_id,
                    round_id=0,
                    parent_prefix_len=r.prompt_length + 1,
                    parent_prefix_hash=token_prefix_hash([*r.prompt_token_ids, bootstraps[i]]),
                    proposal_token_ids=[200 + i],
                    accepted_draft_token_ids=[200 + i],
                    rejected_draft_token_ids=[],
                    accepted_draft_tokens=1,
                    rejected_draft_tokens=0,
                    target_correction_token_ids=[],
                    target_bonus_token_ids=[300 + i],
                    committed_token_ids=[200 + i, 300 + i],
                    logical_target_kv_length=r.prompt_length + 3,
                    logical_draft_kv_length=r.prompt_length + 3,
                    timeline=dict(draft_start_ns=110, state_sync_end_ns=400),
                )
                for i, r in enumerate(definitions)
            ],
        )
    draft_backend.shutdown()
    backend = draft_backend.report()
    backend.update(
        assignment=manifest["assignment"],
        pingpong_work_records=[],
    )
    write_once(run / "draft-backend-report.json", backend)
    timing = [
        dict(
            schema_version=PERFORMANCE_EVENT_SCHEMA,
            event=PERFORMANCE_EVENT,
            consumer={"target": "target-only", "serial": "serial", "pingpong": "dual-batch"}[mode],
            timestamp_ns=100,
            setup_ready_published_ns=90,
            pre_measurement_tp_barrier=True,
            pre_measurement_target_cuda_synchronize=True,
            setup_excluded=True,
        )
    ]
    if not zero and not speculative:
        timing.extend(
            dict(
                schema_version=PERFORMANCE_COMMIT_SCHEMA,
                event="measured-token-commit",
                timestamp_ns=400,
                request_id=r.request_id,
                token_ids=final_tokens[i][1:],
                source="target-tail",
                per_token_cuda_synchronize=False,
            )
            for i, r in enumerate(definitions)
            if i not in terminal
        )
    dump_rows(run / "timing-events.jsonl", timing)
    diagnostics = []
    if not zero:
        for i, r in enumerate(definitions):
            if i in terminal:
                continue
            prefix = [*r.prompt_token_ids, bootstraps[i]]
            diagnostics.append(
                dict(
                    schema_version=DIAGNOSTIC_SCHEMA,
                    request_id=r.request_id,
                    committed_prefix_token_ids=prefix,
                    committed_prefix_sha256=token_prefix_hash(prefix),
                    proposal_token_ids=[],
                    logical_target_kv_length=len(prefix),
                    scheduled_token_count=1,
                    query_length=1,
                    sequence_length=len(prefix),
                    logits_position_mapping=[],
                    position_ids=[len(prefix) - 1],
                    target_input_token_ids=[prefix[-1]],
                    target_forward_start_ns=200,
                    target_forward_end_ns=300,
                    top_raw_logits=[],
                    top_target_logprobs=[],
                    target_verification_shape=[1],
                    attention_backend="synthetic",
                    all_reduce_backend="synthetic",
                    dtype="bfloat16",
                    batch_invariant_requested=True,
                    target_kv_contains_rejected_or_future_tokens=False,
                    causal_attention=True,
                    attention_mask_proof={},
                )
            )
            if speculative:
                diagnostics[-1].update(
                    proposal_token_ids=[200 + i],
                    query_length=2,
                    scheduled_token_count=2,
                    target_input_token_ids=[prefix[-1], 200 + i],
                    position_ids=[len(prefix) - 1, len(prefix)],
                    logits_position_mapping=[
                        dict(
                            proposal_index=0,
                            proposal_token_id=200 + i,
                            target_logits_row=i * 2,
                            predicts_flattened_input_position=i * 2 + 1,
                        )
                    ],
                    attention_mask_proof=dict(
                        causal=True,
                        query_start_range=[2 * i, 2 * i + 2],
                        query_length_matches_scheduler=True,
                        positions_contiguous=True,
                    ),
                )
        dump_rows(run / "target-diagnostics.jsonl", diagnostics)
    if mode == "pingpong":
        states = []
        drafts = []
        for i, r in enumerate(definitions):
            prefix = [*r.prompt_token_ids, *final_tokens[i]]
            transitions = (
                [("BOOTSTRAP", "TERMINAL")]
                if i in terminal
                else [
                    ("BOOTSTRAP", "DRAFT_READY"),
                    ("DRAFT_READY", "TARGET_TAIL_READY"),
                    ("TARGET_TAIL_READY", "VERIFYING"),
                    ("VERIFYING", "COMMITTING"),
                    ("COMMITTING", "TERMINAL"),
                ]
            )
            states.extend(
                dict(
                    request_id=r.request_id,
                    internal_request_id="opaque-" + r.request_id,
                    source_state=a,
                    destination_state=b,
                    timestamp_ns=410 + j,
                    prefix_version=1,
                    committed_prefix_length=len(prefix),
                    committed_prefix_sha256=token_prefix_hash(prefix),
                    reason="synthetic-terminal-contract",
                )
                for j, (a, b) in enumerate(transitions)
            )
            if i not in terminal:
                drafts.append(
                    dict(
                        operation="finish_tail",
                        success=True,
                        request_id=r.request_id,
                        result=dict(committed_token_ids=final_tokens[i][1:]),
                    )
                )
        dump_rows(run / "request-state-events.jsonl", states)
        dump_rows(run / "draft-work-events.jsonl", drafts)
        dump_rows(run / "scheduler-events.jsonl", [])
    return path, run


@pytest.mark.parametrize("mode", ["target", "serial", "pingpong"])
@pytest.mark.parametrize("zero", [False, True])
def test_native_result_entry_tail_and_zero_work(tmp_path, monkeypatch, mode, zero):
    path, run = native_artifacts(tmp_path, monkeypatch, mode, zero)
    report = inspect_run(path, run, mode)
    assert report["valid"], (report["errors"], report.get("error_details"))
    assert report["metrics"]["timed_committed_tokens"] == (0 if zero else 4)
    assert report["work"]["Target_forward_count"] == (0 if zero else 1)
    assert report["work"]["Target_query_tokens"] == (0 if zero else 4)
    assert report["performance_result"] is not zero
    assert report["measurement"]["end_ns"] == 501
    seal_run(run, report)
    assert verify_seal(run) == report
    assert inspect_run(path, run, mode) == report


@pytest.mark.parametrize(
    "fault",
    [
        "null",
        "empty",
        "nonempty",
        "false",
        "accounting",
        "lifecycle",
        "missing-request",
        "missing-forward",
        "early-sync",
        "extra-commit",
        "termination",
    ],
)
def test_native_validator_material_failures_not_empty_serialization(tmp_path, monkeypatch, fault):
    path, run = native_artifacts(tmp_path, monkeypatch)
    raw = read_json(run / "raw.json")
    if fault == "empty":
        raw["errors"] = []
    elif fault == "nonempty":
        raw["errors"] = ["real execution failure"]
    elif fault == "false":
        raw["valid"] = False
    elif fault == "accounting":
        raw["outputs"][0]["generated_tokens"] = 1487
    elif fault == "lifecycle":
        life = read_json(run / "process-lifecycle.json")
        life["cleanup_valid"] = False
        (run / "process-lifecycle.json").write_text(json.dumps(life))
    elif fault == "missing-request":
        raw["outputs"].pop()
    elif fault == "missing-forward":
        (run / "target-diagnostics.jsonl").unlink()
    elif fault == "early-sync":
        raw["phase4b2_final_sync"][0]["final_cuda_synchronize_complete_ns"] = 101
    elif fault == "extra-commit":
        with (run / "timing-events.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        schema_version=PERFORMANCE_COMMIT_SCHEMA,
                        event="measured-token-commit",
                        timestamp_ns=401,
                        request_id="r0",
                        token_ids=[201],
                        per_token_cuda_synchronize=False,
                    )
                )
                + "\n"
            )
    elif fault == "termination":
        raw["outputs"][0]["finish_reason"] = "stop"
    (run / "raw.json").write_text(json.dumps(raw))
    report = inspect_run(path, run, "target")
    assert report["valid"] is (fault in ("null", "empty")), report
    assert report["performance_result"] == report["valid"]


def test_draft_measurement_uses_existing_fences_and_actual_B_Q(tmp_path, monkeypatch):
    from specrhythm.phase4.draft_metrics import DraftMetrics

    metrics = DraftMetrics()
    metrics.forward("proposal", 3, 7)
    metrics.gpu_ms["proposal"] = 0.1
    row = dict(
        purpose="proposal",
        B=3,
        Q=7,
        host_start_ns=200,
        host_launch_end_ns=250,
        gpu_event_ms=0.1,
        cuda_completion_observed_ns=900,
    )
    metrics.s1_forward_records.append(row)
    backend = metrics.snapshot("vllm-batched")
    assert draft_measurement(backend, 100) == [row]
    row["cuda_completion_observed_ns"] = None
    with pytest.raises(ValueError, match="completion"):
        draft_measurement(backend, 100)
    row["cuda_completion_observed_ns"] = 900
    row["Q"] = 8
    with pytest.raises(ValueError, match="accounting"):
        draft_measurement(backend, 100)


def test_offline_command_cannot_dispatch_GPU(monkeypatch, tmp_path):
    from specrhythm.serving import s1_cli

    monkeypatch.setattr(s1_cli, "offline_comparison", lambda *a: dict(valid=True))
    monkeypatch.setattr(s1_cli, "run_attempt", lambda *a: pytest.fail("GPU dispatched"))
    assert s1_cli.main(["compare", "--root", str(tmp_path), "--gate", "G1"]) == 0
    assert list((tmp_path / "G1").glob("offline-comparison-*.json"))


def test_native_serial_speculative_rounds_and_Draft_drain_end(tmp_path, monkeypatch):
    path, run = native_artifacts(tmp_path, monkeypatch, "serial", speculative=True)
    result = inspect_run(path, run, "serial")
    assert result["valid"], (result["errors"], result.get("error_details"))
    assert result["measurement"]["end_ns"] == 900  # Target final sync was only 501.
    assert result["metrics"]["timed_committed_tokens"] == 8
    assert result["work"]["Target_query_tokens"] == 8
    assert result["work"]["Target_physical_forwards"][0]["committed_progress"] == 8
    assert result["work"]["mean_accepted_draft_prefix"] == 1
    assert result["work"]["mean_committed_tokens_per_verification"] == 2
    rounds = [json.loads(line) for line in (run / "round-events.jsonl").read_text().splitlines()]
    rounds[0]["logical_draft_kv_length"] -= 1
    dump_rows(run / "round-events.jsonl", rounds)
    assert not inspect_run(path, run, "serial")["valid"]


@pytest.mark.parametrize("count", [1, 3, 7])
def test_S1_actual_scheduler_assignment_retirement_drain(
    tmp_path, monkeypatch, pingpong_factory, count
):
    from test_phase4_dual_scheduler import proposal_result, retire
    from test_phase4_pingpong import consume

    path, manifest, _ = execution(tmp_path / "input", [request(i) for i in range(count)])
    monkeypatch.setenv(PROFILE_ENV, str(path))
    engine = pingpong_factory(count)
    assert dict(engine.assignment) == manifest["assignment"]
    for rid in engine.assignment:
        row = engine.requests[engine._dual_identity.internal_id(rid)]
        engine._accept_ready_result(proposal_result(rid, prefix=tuple(row.all_token_ids)))
    a = engine.schedule()
    consume(engine, a)
    for rid, cohort in engine.assignment.items():
        if cohort == "A":
            retire(engine, rid)
    b = engine.schedule()
    assert len(b.num_scheduled_tokens) == count // 2
    if count > 1:
        assert engine._dual_events.read()[-1]["pipeline_phase"] == "drain"
        consume(engine, b)
        for rid, cohort in engine.assignment.items():
            if cohort == "B":
                retire(engine, rid)
    assert not engine.schedule().num_scheduled_tokens
    assert dict(engine.assignment) == manifest["assignment"]


def test_resume_cleans_interruption_before_skipping_older_pass(tmp_path, monkeypatch):
    from specrhythm.serving import s1_cli

    path, manifest, _ = execution(tmp_path / "input")
    root = tmp_path / "runs"
    base = root / "G1/0-target"
    good, interrupted = base / "attempt-001", base / "attempt-002"
    good.mkdir(parents=True)
    interrupted.mkdir()
    write_once(good / "process-lifecycle.json", dict(owned_cleanup_completed=True))
    seal_run(good, dict(
        schema_version=RESULT_SCHEMA, **policy_fields(),
        valid=True, execution_sha256=manifest["manifest_sha256"],
    ))
    marker = interrupted / "partial.json"
    marker.write_text("partial GPU evidence retained")
    before = sha256_file(marker)
    cleaned = []
    monkeypatch.setattr(s1_cli, "load_execution", lambda path, **kw: load_execution(path))
    monkeypatch.setattr(s1_cli, "validate_execution_files", lambda *a: None)
    monkeypatch.setattr(s1_cli, "cleanup_attempt", lambda p: cleaned.append(p))
    directory, report = s1_cli.run_attempt(root, "G1", "target", 0, path)
    assert cleaned == [interrupted] and directory == good and report["valid"]
    assert sha256_file(marker) == before and not (base / "attempt-003").exists()


def test_recovery_refuses_unrecorded_root_identity():
    from specrhythm.serving.s1_cli import recover_owner

    with pytest.raises(ValueError, match="root PID/start identity"):
        recover_owner(dict(root_pid=999999999, observed=[]))


def test_supervisor_capacity_failure_records_BLOCKED_and_exit(tmp_path, monkeypatch):
    import os

    from specrhythm.serving import s1_cli
    from specrhythm.serving.common import DataError

    monkeypatch.setattr(s1_cli, "process_table", lambda: {os.getpid(): dict(start_identity="cpu")})

    def blocked(*args):
        raise DataError("insufficient KV", status="BLOCKED")

    monkeypatch.setattr(s1_cli, "gate", blocked)
    assert s1_cli.supervise(tmp_path, "G3", "synthetic-capacity") == 1
    assert read_json(tmp_path / "stage.json")["state"] == "BLOCKED"
    assert read_json(tmp_path / "exit-code-synthetic-capacity.json")["exit_code"] == 1


def test_repeat_round_differences_classified_by_stable_request_round():
    from specrhythm.serving.s1_results import round_differences

    first = [
        dict(request_id="a", round_id=0, prefix_length=10, proposal=[1, 2]),
        dict(request_id="b", round_id=0, prefix_length=20, proposal=[3, 4]),
    ]
    assert round_differences(first, list(reversed(first))) == []
    changed = [{**first[0], "proposal": [2, 3]}, first[1]]
    difference = round_differences(first, changed)
    assert len(difference) == 1 and difference[0]["request_id"] == "a"
    assert difference[0]["classification"] == "same-prefix Draft proposal differs"
