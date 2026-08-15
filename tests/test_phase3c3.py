import hashlib
import json
from pathlib import Path

import pytest

from specrhythm.cli import main
from specrhythm.phase3.engine import DryRunBackend
from specrhythm.phase3.learned_selector import (
    FEATURE_NAMES,
    LearnedShellModel,
    _learned_replay_store,
    select_learned_shell_ranker,
    shell_opportunity_decomposition,
    stratified_request_splits,
)
from specrhythm.phase3.multiround import (
    run_common_prefix_snapshot_stage,
    run_sequential_replay_stage,
    sequential_store,
    snapshot_store,
    summarize_multiround,
    validate_multiround_artifacts,
)
from specrhythm.phase3.phase3c_config import load_phase3c_config
from specrhythm.phase3.r3_workload import build_r3_real_workload, load_r3_workload
from specrhythm.phase3.real_candidate_trace import (
    LabeledTraceRecord,
    TargetFeatureLeakageError,
    run_target_trajectory_stage,
    target_store,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_CONFIG = FIXTURES / "phase3c-config.json"
REPOSITORY = Path(__file__).parents[1]


class QwenAuditTokenizer:
    model_id = "fixture:qwen3"
    tokenizer_fingerprint = "d" * 64
    tokenizer_metadata = {
        "tokenizer_class": "QwenFixtureTokenizer",
        "special_tokens_map": {"eos_token": "<|im_end|>"},
        "model_max_length": 32768,
        "truncation_side": "left",
        "padding_side": "right",
        "chat_template_sha256": "c" * 64,
    }

    def encode(self, prompt):
        return [ord(character) for character in prompt]

    def render_chat(self, user_text):
        return (
            "<|im_start|>user\n"
            + user_text
            + "<|im_end|>\n<|im_start|>assistant\n[THINKING=False]"
        )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config_with_paths(tmp_path, request_count):
    value = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    value["workload"]["request_count"] = request_count
    value["workload"]["arrival_trace"] = str(FIXTURES / "phase3c-arrivals.jsonl")
    value["workload"]["sources"]["code"]["path"] = str(
        FIXTURES / "phase3c-humaneval.jsonl"
    )
    value["workload"]["sources"]["chat"]["path"] = str(
        FIXTURES / "phase3c-sharegpt.jsonl"
    )
    value["workload"]["sources"]["summarization"]["path"] = str(
        FIXTURES / "phase3c-cnndm.jsonl"
    )
    value["candidate_pool"]["phase2_config_path"] = str(
        REPOSITORY / "configs" / "simulator.json"
    )
    path = tmp_path / "phase3c-config.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def corrected_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3c3")
    config_path = _config_with_paths(root, 10)
    config = load_phase3c_config(str(config_path))
    workload = root / "workload.jsonl"
    manifest = root / "manifest.json"
    build_r3_real_workload(
        config,
        output_path=workload,
        manifest_path=manifest,
        command="phase3c3-cpu-fixture",
        git_commit="1" * 40,
    )
    requests = load_r3_workload(workload)
    target = root / "target"
    snapshots = root / "snapshots"
    sequential = root / "sequential"
    run_target_trajectory_stage(
        requests,
        config,
        workload_path=workload,
        output_dir=target,
        resume=False,
        # Share the deterministic dry-run distribution so the tiny fixture contains
        # both target-path positives and alternative-node negatives for training.
        backend=DryRunBackend("draft", 1664),
    )
    run_common_prefix_snapshot_stage(
        requests,
        config,
        workload_path=workload,
        target_dir=target,
        output_dir=snapshots,
        resume=False,
        backend=DryRunBackend("draft", 1664),
    )
    run_sequential_replay_stage(
        requests,
        target_dir=target,
        snapshot_dir=snapshots,
        output_dir=sequential,
        resume=False,
    )
    return {
        "root": root,
        "config": config,
        "workload": workload,
        "manifest": manifest,
        "requests": requests,
        "target": target,
        "snapshots": snapshots,
        "sequential": sequential,
    }


def test_corrected_100_workload_has_60_20_20_and_qwen_chat_audit(tmp_path):
    sources = {
        "code": tmp_path / "humaneval.jsonl",
        "chat": tmp_path / "sharegpt.jsonl",
        "summarization": tmp_path / "cnndm.jsonl",
    }
    _write_jsonl(
        sources["code"],
        [
            {"task_id": f"HumanEval/{index}", "prompt": f"def f_{index}(x):\n    "}
            for index in range(60)
        ],
    )
    _write_jsonl(
        sources["chat"],
        [
            {
                "id": f"chat-{index}",
                "conversations": [{"from": "human", "value": f"Question {index}?"}],
            }
            for index in range(20)
        ],
    )
    _write_jsonl(
        sources["summarization"],
        [
            {"id": f"summary-{index}", "article": f"Document {index}."}
            for index in range(20)
        ],
    )
    arrivals = tmp_path / "arrivals.jsonl"
    _write_jsonl(arrivals, [{"timestamp": index * 10} for index in range(100)])
    value = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    value["workload"]["request_count"] = 100
    value["workload"]["arrival_trace"] = str(arrivals)
    for task, path in sources.items():
        value["workload"]["sources"][task]["path"] = str(path)
    value["candidate_pool"]["phase2_config_path"] = str(
        REPOSITORY / "configs" / "simulator.json"
    )
    config_path = tmp_path / "corrected-100.json"
    config_path.write_text(json.dumps(value), encoding="utf-8")
    workload = tmp_path / "corrected-100.jsonl"
    manifest_path = tmp_path / "corrected-100-manifest.json"
    manifest = build_r3_real_workload(
        load_phase3c_config(str(config_path)),
        output_path=workload,
        manifest_path=manifest_path,
        command="corrected-100-test",
        tokenizer=QwenAuditTokenizer(),
    )
    requests = load_r3_workload(workload)
    assert manifest["task_counts"] == {
        "code": 60,
        "chat": 20,
        "summarization": 20,
    }
    assert len(requests) == 100
    assert manifest["prompt_rendering_audit"]["chat"]["chat_template_applied"]
    assert manifest["prompt_rendering_audit"]["chat"]["enable_thinking"] is False
    assert all(
        "[THINKING=False]" in request.prompt_text
        for request in requests
        if request.task_class == "chat"
    )
    splits = stratified_request_splits(requests, seed=1664)
    assert {name: list(splits.values()).count(name) for name in set(splits.values())} == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }


def test_multiround_validation_and_shell_conservation(corrected_pipeline):
    report = validate_multiround_artifacts(
        corrected_pipeline["requests"],
        workload_path=corrected_pipeline["workload"],
        workload_manifest_path=corrected_pipeline["manifest"],
        target_dir=corrected_pipeline["target"],
        snapshot_dir=corrected_pipeline["snapshots"],
        sequential_dir=corrected_pipeline["sequential"],
        expected_request_count=10,
    )
    assert report["valid"], report["errors"]
    assert all(report["checks"].values())
    cli_validation = corrected_pipeline["root"] / "validation.json"
    assert (
        main(
            [
                "phase3c-multiround-validate",
                "--workload",
                str(corrected_pipeline["workload"]),
                "--workload-manifest",
                str(corrected_pipeline["manifest"]),
                "--target-dir",
                str(corrected_pipeline["target"]),
                "--snapshot-dir",
                str(corrected_pipeline["snapshots"]),
                "--sequential-dir",
                str(corrected_pipeline["sequential"]),
                "--expected-request-count",
                "10",
                "--output",
                str(cli_validation),
            ]
        )
        == 0
    )
    assert json.loads(cli_validation.read_text(encoding="utf-8"))["valid"]
    summary = summarize_multiround(
        snapshot_dir=corrected_pipeline["snapshots"],
        sequential_dir=corrected_pipeline["sequential"],
        source_trace_commit="3" * 40,
    )
    assert {row["pool_ratio"] for row in summary["aggregate_metrics"]} == {
        "1x",
        "2x",
    }
    assert all(
        row["selector"] == "within-request-target-oracle"
        for row in summary["diagnostic_4x_oracle"]
    )
    assert all(
        row["request_bootstrap_95_ci"]
        for row in summary["paired_request_level_deltas"]
    )
    decomposition = shell_opportunity_decomposition(
        snapshot_store(corrected_pipeline["snapshots"]).records()
    )
    for row in decomposition["per_snapshot"]:
        assert row["base_node_count"] + row["shell_node_count"] == 32
        assert (
            row["base_target_path_node_count"]
            + row["shell_target_path_node_count"]
            <= 32
        )
        assert row["shell_nodes_reachable_under_budget_4"] <= row["shell_node_count"]
        for selected in row["selected_nodes"].values():
            assert selected["base"] + selected["shell"] <= 4


def test_request_splits_are_deterministic_stratified_and_leak_free(
    corrected_pipeline,
):
    first = stratified_request_splits(corrected_pipeline["requests"], seed=1664)
    second = stratified_request_splits(corrected_pipeline["requests"], seed=1664)
    assert first == second
    assert set(first) == {request.request_id for request in corrected_pipeline["requests"]}
    assert set(first.values()) == {"train", "validation", "test"}
    assert all(
        isinstance(first[request.request_id], str)
        for request in corrected_pipeline["requests"]
    )


def test_learned_pilot_cli_is_deterministic_prefix_closed_and_resumable(
    corrected_pipeline,
):
    output_dir = corrected_pipeline["root"] / "learned"
    report_path = corrected_pipeline["root"] / "learned-report.json"
    markdown_path = corrected_pipeline["root"] / "learned-report.md"
    arguments = [
        "phase3c-learned-pilot",
        "--workload",
        str(corrected_pipeline["workload"]),
        "--target-dir",
        str(corrected_pipeline["target"]),
        "--snapshot-dir",
        str(corrected_pipeline["snapshots"]),
        "--sequential-dir",
        str(corrected_pipeline["sequential"]),
        "--output-dir",
        str(output_dir),
        "--source-trace-commit",
        "2" * 40,
        "--output",
        str(report_path),
        "--markdown-output",
        str(markdown_path),
    ]
    assert main(arguments) == 0
    immutable_hashes = {
        path.relative_to(output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "stage-summary.json"
    }
    assert main(arguments + ["--resume"]) == 0
    assert immutable_hashes == {
        path.relative_to(output_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "stage-summary.json"
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"]["target_labels_available_at_inference"] is False
    assert report["feature_dataset"]["runtime_and_target_fields_serialized_separately"]
    assert report["decision"]["test_set_used_for_rule_definition"] is False
    model = LearnedShellModel.from_dict(report["model"])
    assert model.feature_names == FEATURE_NAMES
    assert report["artifact_manifest"]["runtime_features_only_at_inference"]
    metric_fields = {
        "auroc",
        "auprc",
        "precision_at_budget_4",
        "recall_at_budget_4",
        "ndcg_at_budget_4",
        "positive_prevalence",
    }
    assert all(
        metric_fields <= set(row)
        for row in report["feature_separability"]
    )
    assert _learned_replay_store(output_dir / "learned-replay").records()

    snapshots = {
        (row["request_id"], row["prefix_position"]): row
        for row in snapshot_store(corrected_pipeline["snapshots"]).records()
    }
    for replay in _learned_replay_store(output_dir / "learned-replay").records():
        for round_row in replay["result"]["rounds"]:
            snapshot = snapshots[(replay["request_id"], round_row["prefix_position"])]
            labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
            by_id = {
                node.runtime_features.stable_node_id: node.runtime_features
                for node in labeled.nodes
            }
            selected = set(round_row["selected_node_ids"])
            assert len(selected) <= 4
            assert all(
                by_id[node_id].parent_id is None
                or by_id[node_id].parent_id in selected
                for node_id in selected
            )


def test_learned_selector_rejects_target_labeled_nodes(corrected_pipeline):
    snapshot = snapshot_store(corrected_pipeline["snapshots"]).records()[0]
    labeled = LabeledTraceRecord.from_dict(snapshot["labeled_trace"])
    model = LearnedShellModel(
        FEATURE_NAMES,
        (0.0,) * len(FEATURE_NAMES),
        (1.0,) * len(FEATURE_NAMES),
        (0.0,) * len(FEATURE_NAMES),
        0.0,
        {},
    )
    with pytest.raises(TargetFeatureLeakageError):
        select_learned_shell_ranker(
            labeled.nodes,
            4,
            model,
            base_node_ids=set(labeled.pool_node_ids["1x"]),
            task_class=labeled.task_class,
            round_index=0,
            remaining_output_length=4,
        )


def test_target_and_replay_artifacts_remain_immutable(corrected_pipeline):
    target_hashes = {
        request.request_id: hashlib.sha256(
            target_store(corrected_pipeline["target"]).path(request.request_id).read_bytes()
        ).hexdigest()
        for request in corrected_pipeline["requests"]
    }
    replay_hashes = {
        row["request_id"]: hashlib.sha256(
            sequential_store(corrected_pipeline["sequential"])
            .path(row["request_id"])
            .read_bytes()
        ).hexdigest()
        for row in sequential_store(corrected_pipeline["sequential"]).records()
    }
    run_sequential_replay_stage(
        corrected_pipeline["requests"],
        target_dir=corrected_pipeline["target"],
        snapshot_dir=corrected_pipeline["snapshots"],
        output_dir=corrected_pipeline["sequential"],
        resume=True,
    )
    assert target_hashes == {
        request.request_id: hashlib.sha256(
            target_store(corrected_pipeline["target"]).path(request.request_id).read_bytes()
        ).hexdigest()
        for request in corrected_pipeline["requests"]
    }
    assert replay_hashes == {
        row["request_id"]: hashlib.sha256(
            sequential_store(corrected_pipeline["sequential"])
            .path(row["request_id"])
            .read_bytes()
        ).hexdigest()
        for row in sequential_store(corrected_pipeline["sequential"]).records()
    }


def test_phase3c3_new_modules_keep_python39_annotation_syntax():
    for relative in (
        "src/specrhythm/phase3/learned_selector.py",
        "src/specrhythm/phase3/multiround.py",
    ):
        text = (REPOSITORY / relative).read_text(encoding="utf-8")
        assert " | None" not in text
