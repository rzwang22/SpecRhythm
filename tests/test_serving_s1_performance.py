"""S1-P synthetic CPU contracts; no numerical/GPU/performance qualification."""

import copy
import inspect
import json
import shutil
from functools import partial

import pytest
from test_serving_s1 import execution, native_artifacts, outputs, request, synthetic_result

from specrhythm.phase4.stock_vllm import run_stock_smoke
from specrhythm.serving.common import digest, read_json
from specrhythm.serving.s1_cli import previous_completed, run_plan, start
from specrhythm.serving.s1_policy import (
    COMPARISON_SCHEMA,
    G0_SCHEMA,
    policy_fields,
    validate_policy,
)
from specrhythm.serving.s1_results import (
    compare_results,
    inspect_run,
    seal_run,
    verify_seal,
    workload_metrics,
)
from specrhythm.serving.s1_runtime import qualify_raw_target
from specrhythm.serving.s1_workload import load_execution, pending_reference_comparison, write_once


def test_native_three_modes_different_outputs_and_repeats_are_valid(tmp_path, monkeypatch):
    from specrhythm.serving import s1_results

    # If the S1-P comparator accidentally reintroduces output comparison, fail immediately.
    def forbidden(*args):
        pytest.fail("S1-P must not compare independent tokens or round semantics")

    monkeypatch.setattr(s1_results, "first_divergence", forbidden)
    monkeypatch.setattr(s1_results, "round_differences", forbidden)
    results = {m: [] for m in ("target", "serial", "pingpong")}
    for index, (mode, repeat) in enumerate(run_plan("G1")):
        path, directory = native_artifacts(
            tmp_path / f"{mode}-{repeat}",
            monkeypatch,
            mode,
            token_offset=index * 10,
            setup_terminal=tuple(range(index)),
            eos_finish=(3,),
        )
        result = inspect_run(path, directory, mode)
        assert result["valid"], result
        assert result["metrics"]["timed_committed_tokens"] == 4 - index
        assert result["metrics"]["untimed_bootstrap_tokens"] == 4
        assert result["metrics"]["setup_terminal_requests"] == index
        seal_run(directory, result)
        assert verify_seal(directory) == result
        results[mode].append(result)
    compared = compare_results(results)
    assert compared["valid"], compared["errors"]
    assert compared["cross_run_output_comparison"] == dict(performed=False, status="NOT_REQUIRED")
    assert compared["repeatability"]["exact_tokens_and_termination"] is None
    assert compared["matched_work"]["exact_timed_output_equal"] is None
    assert compared["matched_work"]["equal_timed_token_counts"] is False
    assert compared["performance_ratios"]["pingpong"]["median_throughput_ratio"] == pytest.approx(
        (2 + 1) / 2 / 4
    )
    assert compared["performance_ratios"]["pingpong"]["throughput_above_target"] is False
    assert compared["end_to_end_improvement_claim"] is False


def test_actual_rates_and_repeat_statistics_do_not_assume_matched_work():
    results = {m: [] for m in ("target", "serial", "pingpong")}
    for mode, samples in zip(results, (((4, 2), (8, 4)), ((3, 1), (8, 2)), ((2, 4), (4, 2)))):
        for repeat, (timed, seconds) in enumerate(samples):
            result = synthetic_result(mode)
            row = result["requests"][0]
            row.update(
                generated_token_ids=list(
                    range(100 * (repeat + 1), 100 * (repeat + 1) + timed + 1)
                ),
                generated_tokens=timed + 1,
                timed_committed_tokens=timed,
            )
            result["metrics"] = workload_metrics([row], 1, 1 + seconds * 10**9)
            results[mode].append(result)
    report = compare_results(results)
    assert report["valid"]
    assert report["metrics"]["serial"]["throughput_tokens_per_second"] == dict(
        raw=[3, 4],
        median=3.5,
        population_stdev=0.5,
    )
    assert report["performance_ratios"]["serial"]["median_throughput_ratio"] == 1.75
    assert report["performance_ratios"]["serial"]["median_makespan_ratio"] == 2
    assert report["performance_ratios"]["serial"]["target_timed_tokens_by_repeat"] == [4, 8]
    assert report["performance_ratios"]["serial"]["mode_timed_tokens_by_repeat"] == [3, 8]


@pytest.mark.parametrize(
    "fault",
    [
        "bootstrap",
        "duplicate",
        "eos",
        "exit",
        "cleanup",
        "stale",
        "kv",
        "commit",
        "prefix",
    ],
)
def test_internal_failures_still_block_s1p(tmp_path, monkeypatch, fault):
    from test_serving_s1 import dump_rows

    from specrhythm.serving.s1_results import jsonl

    path, directory = native_artifacts(tmp_path, monkeypatch, "serial", speculative=True)
    raw = read_json(directory / "raw.json")
    rounds = jsonl(directory, "round-events")
    if fault == "bootstrap":
        raw["outputs"][0]["generated_token_ids"][0] = 8
    elif fault == "duplicate":
        raw["outputs"].append(copy.deepcopy(raw["outputs"][0]))
    elif fault == "eos":
        raw["outputs"][0]["generated_token_ids"][1] = 999
    elif fault == "exit":
        (directory / "exit-code.json").write_text(
            json.dumps(
                dict(
                    effective_exit_code=7,
                    coordinator_exit_code=7,
                )
            )
        )
    elif fault == "cleanup":
        lifecycle = read_json(directory / "process-lifecycle.json")
        lifecycle["remaining_owned_pids"] = [123]
        (directory / "process-lifecycle.json").write_text(json.dumps(lifecycle))
    elif fault == "stale":
        rounds[0]["prefix_version"] = 0
    elif fault == "kv":
        rounds[0]["logical_draft_kv_length"] += 1
    elif fault == "commit":
        rounds[0]["committed_token_ids"][-1] = 8
    elif fault == "prefix":
        rounds[0]["parent_prefix_hash"] = "a" * 64
    (directory / "raw.json").write_text(json.dumps(raw))
    dump_rows(directory / "round-events.jsonl", rounds)
    report = inspect_run(path, directory, "serial")
    assert not report["valid"] and report["errors"]
    assert not report["performance_result"]
    if fault in ("exit", "cleanup"):
        assert not report["checks"]["lifecycle"]["valid"]


def test_optional_raw_target_differences_do_not_mask_internal_failure():
    rows = [request(0, 3)]
    raw = dict(
        runs=[outputs(rows, [[1, 2, 3]]), outputs(rows, [[4, 999]])],
        repeated_run_deterministic=False,
    )
    report = qualify_raw_target(raw, rows, [999])
    assert report["valid"]
    assert report["repeated_run_deterministic"] is False
    assert report["cross_run_output_comparison"]["status"] == "NOT_REQUIRED"
    raw["runs"][1][0]["generated_tokens"] = 3
    assert not qualify_raw_target(raw, rows, [999])["valid"]
    assert not qualify_raw_target({**raw, "valid": False}, rows, [999])["valid"]
    assert (
        inspect.signature(run_stock_smoke).parameters["compare_repeated_outputs"].default is True
    )
    pending = pending_reference_comparison()
    assert pending["status"] == "NOT_REQUIRED" and pending["valid"] is None
    assert pending["all_sequences_equal"] is None


@pytest.mark.parametrize("location", ["manifest", "result", "seal", "start", "previous-gate"])
def test_old_policy_rejected_without_mutating_old_artifacts(tmp_path, monkeypatch, location):
    from specrhythm.serving import s1_cli

    path, manifest, _ = execution(tmp_path / "input")
    run = tmp_path / "attempt-001"
    run.mkdir()
    report = synthetic_result()
    seal_run(run, report)
    if location == "manifest":
        old = {k: v for k, v in manifest.items() if k not in policy_fields()}
        old["schema_version"] = "specrhythm.s1-execution.v1"
        old["manifest_sha256"] = digest({k: v for k, v in old.items() if k != "manifest_sha256"})
        path.write_text(json.dumps(old))
        action = partial(load_execution, path)
    elif location == "result":
        report.pop("acceptance_policy")
        # Directly validate a fully hash-sealed old-policy report, not a checksum corruption.
        (run / "result.json").write_text(json.dumps(report))
        from specrhythm.phase4.manifest import sha256_file

        seal = read_json(run / "seal.json")
        seal["files"]["result.json"] = sha256_file(run / "result.json")
        (run / "seal.json").write_text(json.dumps(seal))
        action = partial(previous_completed, [run], "a" * 64)
    elif location == "seal":
        seal = read_json(run / "seal.json")["files"]
        (run / "seal.json").write_text(json.dumps(seal))
        action = partial(verify_seal, run)
    elif location == "start":
        write_once(tmp_path / "g0.json", {"valid": True})
        action = partial(start, tmp_path, "G1")
    else:
        write_once(tmp_path / "G1/comparison.json", {"valid": True})
        action = partial(s1_cli.require_previous_gate, tmp_path, "G2")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="policy mismatch"):
        action()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_g1_gate_offline_resume_and_previous_gate_allow_output_variation(tmp_path, monkeypatch):
    from specrhythm.serving import s1_cli

    root = tmp_path / "root"
    records = {}
    for index, (mode, repeat) in enumerate(run_plan("G1")):
        path, source = native_artifacts(
            tmp_path / f"source-{index}",
            monkeypatch,
            mode,
            token_offset=10 * index,
            setup_terminal=tuple(range(index)),
            eos_finish=(3,),
        )
        if index == 0:
            shutil.copytree(path.parent, root / "s1-smoke4")
        dest = root / "G1" / f"{repeat}-{mode}" / "attempt-001"
        shutil.copytree(source, dest)
        report = inspect_run(root / "s1-smoke4/execution-manifest.json", dest, mode)
        assert report["valid"], report
        seal_run(dest, report)
        records[(mode, repeat)] = (dest, report)
    manifest = read_json(root / "s1-smoke4/execution-manifest.json")
    write_once(
        root / "g0.json",
        dict(
            schema_version=G0_SCHEMA,
            **policy_fields(),
            valid=True,
            execution=manifest["execution"],
        ),
    )
    calls = []

    def fake_attempt(root, gate, mode, repeat, manifest_path):
        calls.append((mode, repeat))
        return records[(mode, repeat)]

    monkeypatch.setattr(s1_cli, "load_execution", lambda path, **kw: load_execution(path))
    monkeypatch.setattr(s1_cli, "run_attempt", fake_attempt)
    monkeypatch.setattr(s1_cli, "gate_capacity", lambda *args: None)
    assert s1_cli.gate(root, "G1") == 0
    assert calls == run_plan("G1") and all(mode != "raw-target" for mode, _ in calls)
    report = read_json(root / "G1/comparison.json")
    validate_policy(report, COMPARISON_SCHEMA)
    assert s1_cli.offline_comparison(root, "G1") == report
    s1_cli.require_previous_gate(root, "G2")
    before = (root / "G1/comparison.json").read_bytes()
    assert s1_cli.gate(root, "G1") == 0
    assert (root / "G1/comparison.json").read_bytes() == before
    for directory, result in records.values():
        assert previous_completed([directory], manifest["manifest_sha256"])[1] == result


@pytest.mark.parametrize("fault", ["policy", "schema", "sampling", "nan", "errors"])
def test_comparison_rejects_mixed_policy_config_and_invalid_metrics(fault):
    reports = {m: [synthetic_result(m)] for m in ("target", "serial", "pingpong")}
    bad = reports["serial"][0]
    if fault == "policy":
        bad["cross_mode_token_equality_required"] = True
    elif fault == "schema":
        bad["schema_version"] = "specrhythm.s1-result.v1"
    elif fault == "sampling":
        bad["effective_runtime"] = {"sampling": {"ignore_eos": True}}
    elif fault == "nan":
        bad["metrics"]["throughput_tokens_per_second"] = float("nan")
    else:
        bad["errors"] = ["runtime failure"]
    if fault in ("policy", "schema"):
        with pytest.raises(ValueError):
            compare_results(reports)
    else:
        result = compare_results(reports)
        assert not result["valid"] and result["performance_ratios"] is None
