import ast
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from specrhythm.cli import main
from specrhythm.phase3.benchmark import run_latency_benchmark
from specrhythm.phase3.config import Phase3Config, load_phase3_config
from specrhythm.phase3.engine import DryRunBackend, EngineUnavailableError
from specrhythm.phase3.probe import probe_gpu_environment
from specrhythm.phase3.runner import (
    PromptRequest,
    generate_candidate_forest,
    run_phase3,
    select_candidates,
)
from specrhythm.phase3.tp import validate_tp_compatibility
from specrhythm.phase3.trace import (
    CandidateNodeRecord,
    CycleAccounting,
    RealTraceRecord,
    RequestCycle,
    TargetOutcomeRecord,
    TraceStore,
)

ROOT = Path(__file__).parents[1]
TRACE_CONFIG = ROOT / "configs" / "phase3_trace_1d4v.yaml"
TRACE_CONFIG_1D2V = ROOT / "configs" / "phase3_trace_1d2v.yaml"
LATENCY_CONFIG_1D2V = ROOT / "configs" / "phase3_latency_1d2v.yaml"
PROMPTS = ROOT / "configs" / "phase3-smoke-prompts.jsonl"


def _config(**overrides):
    config = load_phase3_config(str(TRACE_CONFIG)).with_overrides(backend="dry-run")
    return config.with_overrides(**overrides)


def test_three_gpu_configs_use_draft_tp1_and_target_tp2():
    for path in (TRACE_CONFIG_1D2V, LATENCY_CONFIG_1D2V):
        config = load_phase3_config(str(path))
        assert config.draft.gpu_ids == (0,)
        assert config.draft.tp_size == 1
        assert config.target.gpu_ids == (1, 2)
        assert config.target.tp_size == 2
        assert config.backend == "transformers"


def _record(request_id="r0", cycle_id=0):
    request = RequestCycle(
        request_id=request_id,
        cycle_id=cycle_id,
        prompt_length=2,
        context_length=2,
        generated_tokens=(),
        slo_class="40ms",
        draft_model="draft",
        target_model="target",
        random_seed=1664,
        sampling_configuration={"do_sample": False},
        mode="serial",
    )
    node = CandidateNodeRecord(
        stable_node_id="n0",
        parent_id=None,
        depth=1,
        token_id=7,
        local_probability=0.8,
        path_probability=0.8,
        draft_logit=2.0,
        entropy=0.5,
        top1_top2_margin=1.0,
        sibling_rank=0,
        prefix_closed=True,
        selected_for_verification=True,
    )
    outcome = TargetOutcomeRecord("n0", 7, -0.1, True, True, True, 1)
    return RealTraceRecord(
        request=request,
        candidate_nodes=(node,),
        target_outcomes=(outcome,),
        accounting=CycleAccounting(1, 1, 1, 1, 1, 1),
        root_target_token_id=9,
        committed_token_ids=(7, 9),
        request_finished=True,
    )


def test_gpu_probe_reports_explicit_no_cuda(monkeypatch, tmp_path):
    from specrhythm.phase3 import probe

    monkeypatch.setattr(
        probe,
        "_torch_metadata",
        lambda errors: (
            {
                "pytorch_version": None,
                "cuda_runtime": None,
                "nccl_version": None,
                "torch_cuda_available": False,
            },
            [],
        ),
    )

    def command(argv, cwd=None):
        if argv[:2] == ["git", "rev-parse"]:
            return "a" * 40, ""
        return None, "nvidia-smi not found"

    monkeypatch.setattr(probe, "_command", command)
    report = probe_gpu_environment(repo=tmp_path)
    assert report["available"] is False
    assert report["gpu_count"] == 0
    assert report["cuda_runtime"] is None
    assert any("nvidia-smi" in error for error in report["errors"])
    output = tmp_path / "probe.json"
    assert main(["gpu-probe", "--output", str(output)]) == 2
    assert main(["gpu-probe", "--output", str(output), "--allow-unavailable"]) == 0
    assert json.loads(output.read_text())["available"] is False


def test_tp_compatibility_accepts_1_2_4_and_rejects_3_without_surgery():
    config = {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "model_type": "llama",
    }
    report = validate_tp_compatibility(config)
    by_tp = {row["tp_size"]: row for row in report["results"]}
    assert by_tp[1]["supported"]
    assert by_tp[2]["supported"]
    assert not by_tp[3]["supported"]
    assert "num_attention_heads" in by_tp[3]["reason"]
    assert by_tp[4]["supported"]
    assert report["model_surgery_allowed"] is False


def test_unknown_engine_tp_plan_is_conservatively_rejected():
    config = {
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "vocab_size": 30000,
        "model_type": "unlisted_model",
    }
    report = validate_tp_compatibility(config, (1, 2))
    assert report["results"][0]["supported"]
    assert not report["results"][1]["supported"]
    assert "no declared native TP plan" in report["results"][1]["reason"]


def test_real_trace_round_trip_and_selector_target_isolation():
    record = _record()
    restored = RealTraceRecord.from_dict(record.to_dict())
    assert restored == record
    view = restored.selector_view()
    assert not hasattr(view, "target_outcomes")
    assert not hasattr(view.candidates[0], "target_token_id")
    selector_json = json.dumps(
        {
            "request": view.request.to_dict(),
            "candidates": [candidate.__dict__ for candidate in view.candidates],
        }
    )
    assert "accepted_prefix_length" not in selector_json
    assert "target_log_probability" not in selector_json


def test_checkpoint_resume_never_overwrites_completed_cycle(tmp_path):
    store = TraceStore(tmp_path / "trace")
    record = _record()
    assert store.write(record)
    original = store.record_path("r0", 0).read_bytes()
    assert store.write(record) is False
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        store.write(replace(record, committed_token_ids=(7, 10)))
    assert store.record_path("r0", 0).read_bytes() == original
    cycle, generated, finished = store.resume_state("r0")
    assert cycle == 1
    assert generated == (7, 9)
    assert finished


def test_selected_candidate_forest_is_prefix_closed_and_deterministic():
    backend = DryRunBackend("draft", 1664)
    first = generate_candidate_forest(
        backend,
        request_id="r",
        cycle_id=2,
        context=[1, 2],
        search_pool_size=12,
        width=3,
    )
    second = generate_candidate_forest(
        backend,
        request_id="r",
        cycle_id=2,
        context=[1, 2],
        search_pool_size=12,
        width=3,
    )
    assert first == second
    selected = select_candidates(first, 7)
    selected_ids = {node.stable_node_id for node in selected if node.selected_for_verification}
    assert len(selected_ids) == 7
    for node in selected:
        if node.selected_for_verification and node.parent_id is not None:
            assert node.parent_id in selected_ids


def test_trace_rejects_non_prefix_closed_verification():
    parent = replace(
        _record().candidate_nodes[0],
        stable_node_id="parent",
        selected_for_verification=False,
    )
    child = replace(
        parent,
        stable_node_id="child",
        parent_id="parent",
        depth=2,
        path_probability=0.5,
        selected_for_verification=True,
    )
    with pytest.raises(ValueError, match="not prefix closed"):
        RealTraceRecord(
            request=_record().request,
            candidate_nodes=(parent, child),
            target_outcomes=(),
            accounting=CycleAccounting(1, 2, 1, 0, 0, 0),
            root_target_token_id=None,
            committed_token_ids=(),
            request_finished=False,
        )


def test_candidate_token_root_accounting_is_conserved():
    record = _record()
    assert record.accounting.request_roots == 1
    assert record.accounting.search_pool_nodes == len(record.candidate_nodes)
    assert record.accounting.verified_candidate_nodes == sum(
        node.selected_for_verification for node in record.candidate_nodes
    )
    assert len(record.committed_token_ids) == (
        record.accounting.committed_candidate_tokens
        + record.accounting.committed_target_tokens
    )
    with pytest.raises(ValueError, match="commit exactly once"):
        CycleAccounting(1, 1, 1, 1, 0, 1)


def test_serial_runner_accounts_for_accepted_candidates(tmp_path):
    backend = DryRunBackend("shared-model", 1664)
    report = run_phase3(
        [PromptRequest("r", "shared tokenizer and logits", "40ms", 4)],
        _config(search_pool_size=8, candidate_budget=4, max_new_tokens=4),
        mode="serial",
        output_dir=tmp_path / "accepted",
        resume=False,
        draft_backend=backend,
        target_backend=backend,
    )
    assert report["accounting"]["accepted_candidate_nodes"] > 0
    assert report["accounting"]["committed_candidate_tokens"] == report[
        "accounting"
    ]["accepted_candidate_nodes"]
    assert report["accounting"]["committed_target_tokens"] > 0


def test_dry_run_resume_and_target_only_semantic_equivalence(tmp_path):
    requests = [PromptRequest("r", "hello deterministic world", "40ms", 7)]
    config = _config(search_pool_size=8, candidate_budget=4, max_new_tokens=7)
    serial_dir = tmp_path / "serial"
    target_dir = tmp_path / "target"
    first = run_phase3(
        requests,
        config,
        mode="serial",
        output_dir=serial_dir,
        resume=False,
        cycle_limit=2,
    )
    assert TraceStore(serial_dir).resume_state("r")[2] is False
    resumed = run_phase3(
        requests,
        config,
        mode="serial",
        output_dir=serial_dir,
        resume=True,
    )
    run_phase3(
        requests,
        config,
        mode="target-only",
        output_dir=target_dir,
        resume=False,
    )
    assert first["new_records"] > 0
    assert resumed["new_records"] > 0
    completed_resume = run_phase3(
        requests,
        config,
        mode="serial",
        output_dir=serial_dir,
        resume=True,
    )
    assert completed_resume["new_records"] == 0
    assert TraceStore(serial_dir).resume_state("r")[1] == TraceStore(
        target_dir
    ).resume_state("r")[1]
    assert TraceStore(serial_dir).validate()["valid"]


def test_phase3_dry_run_cli_validate_and_summarize(tmp_path):
    serial_dir = tmp_path / "serial"
    target_dir = tmp_path / "target"
    environment = tmp_path / "environment.json"
    environment.write_text('{"available":false}\n')
    for mode, output in (("serial", serial_dir), ("target-only", target_dir)):
        assert (
            main(
                [
                    "phase3-run",
                    "--config",
                    str(TRACE_CONFIG),
                    "--backend",
                    "dry-run",
                    "--mode",
                    mode,
                    "--input",
                    str(PROMPTS),
                    "--output-dir",
                    str(output),
                    "--max-new-tokens",
                    "4",
                    "--search-pool-size",
                    "8",
                    "--candidate-budget",
                    "4",
                    "--environment-metadata",
                    str(environment),
                ]
            )
            == 0
        )
    validation = tmp_path / "validation.json"
    assert (
        main(
            [
                "phase3-validate",
                "--trace-dir",
                str(serial_dir),
                "--target-only-dir",
                str(target_dir),
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    assert json.loads(validation.read_text())["target_only_semantic_equivalence"]
    summary = tmp_path / "summary.json"
    trace = tmp_path / "trace.jsonl"
    assert (
        main(
            [
                "phase3-summarize",
                "--trace-dir",
                str(serial_dir),
                "--trace-output",
                str(trace),
                "--output",
                str(summary),
            ]
        )
        == 0
    )
    assert json.loads(summary.read_text())["trace_jsonl_sha256"]
    manifest = json.loads((serial_dir / "manifest.json").read_text())
    assert manifest["gpu_experiment_executed"] is False
    assert manifest["environment_metadata_file"] == "environment.json"
    assert len(manifest["environment_metadata_sha256"]) == 64


def test_phase3_sources_parse_as_python39():
    for path in (ROOT / "src" / "specrhythm" / "phase3").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))


def test_latency_benchmark_forbids_dry_run_fallback():
    with pytest.raises(EngineUnavailableError, match="dry-run timing is forbidden"):
        run_latency_benchmark(_config(), ("draft",))


@pytest.mark.gpu
def test_real_cuda_backend_is_opt_in_only():
    if os.environ.get("SR_RUN_GPU_TESTS") != "1":
        pytest.skip("set SR_RUN_GPU_TESTS=1 on an NVIDIA host")
    config: Phase3Config = load_phase3_config(str(TRACE_CONFIG))
    from specrhythm.phase3.engine import create_backend

    try:
        backend = create_backend("transformers", config.draft, config.random_seed)
    except EngineUnavailableError as error:
        pytest.fail(str(error))
    backend.close()
