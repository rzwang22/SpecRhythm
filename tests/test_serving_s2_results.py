"""S2 full offline validator using producer-shaped native runtime artifacts."""

import copy
import json

import pytest
from test_serving_s1 import native_artifacts
from test_serving_s2 import profile

from specrhythm.phase4.decode_ready import load_decode_ready_manifest
from specrhythm.serving.common import read_json
from specrhythm.serving.runtime_profile import load_s2
from specrhythm.serving.s1_workload import load_execution, write_once
from specrhythm.serving.s2_clock import ServingClock
from specrhythm.serving.s2_plan import RATIO, poisson_trace, sealed, slo_policy
from specrhythm.serving.s2_pool import prefix_record
from specrhythm.serving.s2_results import qualify, seal_result, verify_result


def fixture(tmp_path, monkeypatch, mode="target", offset=0):
    old, directory = native_artifacts(tmp_path / "native", monkeypatch, mode, token_offset=offset)
    previous, definitions = load_execution(old)
    ready = load_decode_ready_manifest(read_json(directory / "decode-ready-manifest.json"))
    execution = {**previous["execution"], "git_commit": ready.specrhythm_git_commit}
    path, value, _ = profile(tmp_path / "s2", rows=definitions, execution=execution)
    trace = poisson_trace(value["request_ids"], 1e9)
    value = sealed({**{k: v for k, v in value.items() if k != "sha256"}, "trace": trace})
    path.write_text(json.dumps(value))
    raw = read_json(directory / "raw.json")
    boot = {
        r["request_id"]: {
            "token": r["generated_token_ids"][0],
            "terminal": False,
            "finish_reason": None,
        }
        for r in raw["outputs"]
    }
    clock = ServingClock(definitions, trace, boot, mode=mode)
    clock.start(100, threaded=False)
    clock.observe(110)
    clock.admit(120)
    for r in raw["outputs"]:
        clock.commit(
            r["request_id"],
            r["generated_token_ids"][1:],
            400,
            finished=True,
            finish_reason=r["finish_reason"],
        )
        clock.released([r["request_id"]], 401)
    initial = {
        r.request_id: prefix_record(r.prompt_token_ids, r.prompt_length, [[i + 1]])
        for i, r in enumerate(definitions)
    }
    pool = {
        "initial": initial,
        "initial_requests": 4,
        "initial_blocks": 4,
        "peak_blocks": 4,
        "resident_checks": 2,
    }
    runtime = {
        "start_ns": 100,
        "end_ns": 1000,
        "requests": list(clock.rows.values()),
        "events": clock.events,
        "target_steps": [],
        "target_pool_final": pool,
        "target_final_sync": [{"rank": 0, "timestamp_ns": 500}, {"rank": 1, "timestamp_ns": 501}],
        "draft_shutdown": {"shutdown": True},
        "target_final_memory": [],
    }
    write_once(directory / "runtime.json", runtime)
    write_once(
        directory / "arrival-output-events.json",
        {"events": clock.events, "requests": runtime["requests"], "failure": None},
    )
    write_once(
        directory / "resident-pool.json",
        {
            "selected_requests": 4,
            "prefill_setup_ns": 40,
            "target_rank_initial_memory": [],
            "target": pool,
            "draft": {"rows": initial},
        },
    )
    backend = read_json(directory / "draft-backend-report.json")
    backend.update(
        s2_live_requests_before_shutdown=0,
        s2_resident_pool=pool,
        s2_memory_final={},
        s2_work_records=[],
    )
    (directory / "draft-backend-report.json").write_text(json.dumps(backend))
    exits = read_json(directory / "exit-code.json")
    exits["draft_exit_code"] = 0
    (directory / "exit-code.json").write_text(json.dumps(exits))
    if mode == "pingpong":
        write_once(directory / "plugin-report.json", {"sampled_row_tp_consensus": True})
    return path, directory, slo_policy(dict.fromkeys(RATIO, 100))


@pytest.mark.parametrize("mode", ("target", "serial", "pingpong"))
def test_full_native_S2_validator_passes_and_requalifies_identically(tmp_path, monkeypatch, mode):
    path, directory, policy = fixture(tmp_path, monkeypatch, mode)
    value = qualify(path, directory, mode, policy)
    assert value["valid"], (value["errors"], value.get("primary_error"))
    assert value["metrics"]["by_task"]["overall"]["timed_tokens"] == 4
    assert value["cross_run_token_length_EOS_round_equality"] == "NOT_REQUIRED"
    assert value["overlap_ms"] == 0
    seal_result(directory, value)
    assert verify_result(directory) == value
    assert qualify(path, directory, mode, policy) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_exit_code", 2),
        ("cleanup_valid", False),
        ("s2_live_requests_before_shutdown", 1),
        ("execution_failed", True),
    ],
)
def test_full_S2_internal_failures_block_with_artifact_details(
    tmp_path, monkeypatch, field, value
):
    path, directory, policy = fixture(tmp_path, monkeypatch)
    name = (
        "exit-code.json"
        if field == "effective_exit_code"
        else "process-lifecycle.json"
        if field == "cleanup_valid"
        else "draft-backend-report.json"
    )
    artifact = directory / name
    data = read_json(artifact)
    data[field] = value
    artifact.write_text(json.dumps(data))
    result = qualify(path, directory, "target", policy)
    assert not result["valid"]
    assert result["primary_error"]["artifact"] == str(artifact)
    assert "expected" in result["primary_error"] and "actual" in result["primary_error"]


def test_S2_different_native_outputs_still_qualify(tmp_path, monkeypatch):
    values = []
    for index in (0, 10):
        path, directory, policy = fixture(tmp_path / str(index), monkeypatch, offset=index)
        values.append(qualify(path, directory, "target", policy))
    assert all(v["valid"] for v in values)


def test_native_commit_disagreement_blocks_even_if_ledger_is_self_consistent(
    tmp_path, monkeypatch
):
    path, directory, policy = fixture(tmp_path, monkeypatch)
    runtime = read_json(directory / "runtime.json")
    runtime["requests"][0]["generated_token_ids"][-1] = 345
    runtime["requests"][0]["commits"][0]["token_ids"][-1] = 345
    (directory / "runtime.json").write_text(json.dumps(runtime))
    observed = read_json(directory / "arrival-output-events.json")
    observed["requests"] = copy.deepcopy(runtime["requests"])
    (directory / "arrival-output-events.json").write_text(json.dumps(observed))
    value = qualify(path, directory, "target", policy)
    assert not value["valid"] and "native worker commits" in value["errors"][0]


def test_population_uses_event_time_when_arrival_thread_publishes_first(tmp_path, monkeypatch):
    from specrhythm.serving.s2_results import validate_population

    path, directory, _ = fixture(tmp_path, monkeypatch)
    runtime = read_json(directory / "runtime.json")
    _, definitions = load_s2(str(path))
    expected = validate_population(runtime, definitions, 128)
    # Model independent publisher interleaving: append order across requests differs
    # from the already captured GPU-output and arrival timestamps.
    runtime["events"].sort(key=lambda e: -e["timestamp_ns"])
    assert validate_population(runtime, definitions, 128) == expected
