import hashlib
import json
from pathlib import Path

import pytest

from specrhythm.cli import main
from specrhythm.phase3.engine import DryRunBackend
from specrhythm.phase3.multiround import (
    multiround_headroom,
    run_common_prefix_snapshot_stage,
    run_sequential_replay_stage,
    sequential_store,
    snapshot_store,
    summarize_multiround,
)
from specrhythm.phase3.phase3c_config import load_phase3c_config
from specrhythm.phase3.r3_workload import (
    build_r3_real_workload,
    load_r3_workload,
)
from specrhythm.phase3.real_candidate_trace import (
    labeled_store,
    run_draft_forest_stage,
    run_label_join_stage,
    run_target_trajectory_stage,
    target_store,
)
from specrhythm.phase3.selector_diagnosis import (
    COVERAGE_DEFINITION_VERSION,
    POOL_ORDER,
    TARGET_BLIND_SELECTORS,
    _selection_stability,
    coverage_metrics,
    run_selector_replay_stage,
    selector_store,
    stratified_bootstrap_ci,
    summarize_selector_diagnosis,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = FIXTURES / "phase3c-config.json"


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


@pytest.fixture()
def phase3c2_pipeline(tmp_path):
    config = load_phase3c_config(str(CONFIG))
    workload = tmp_path / "workload.jsonl"
    manifest = tmp_path / "manifest.json"
    build_r3_real_workload(
        config,
        output_path=workload,
        manifest_path=manifest,
        command="phase3c2-fixture",
        git_commit="1" * 40,
    )
    requests = load_r3_workload(workload)
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    labeled = tmp_path / "labeled"
    selectors = tmp_path / "selectors"
    run_draft_forest_stage(
        requests,
        config,
        workload_path=workload,
        output_dir=draft,
        resume=False,
        backend=DryRunBackend("draft", 1664),
    )
    run_target_trajectory_stage(
        requests,
        config,
        workload_path=workload,
        output_dir=target,
        resume=False,
        backend=DryRunBackend("target", 1664),
    )
    run_label_join_stage(
        requests,
        forest_dir=draft,
        target_dir=target,
        output_dir=labeled,
        resume=False,
    )
    run_selector_replay_stage(config, labeled_dir=labeled, output_dir=selectors, resume=False)
    return {
        "config": config,
        "workload": workload,
        "manifest": manifest,
        "requests": requests,
        "draft": draft,
        "target": target,
        "labeled": labeled,
        "selectors": selectors,
    }


def test_prompt_audit_applies_qwen_template_and_disables_thinking(tmp_path):
    config = load_phase3c_config(str(CONFIG))
    workload = tmp_path / "workload.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_r3_real_workload(
        config,
        output_path=workload,
        manifest_path=manifest_path,
        command="prompt-audit",
        tokenizer=QwenAuditTokenizer(),
    )
    audit = manifest["prompt_rendering_audit"]
    assert audit["code"]["instruction"].startswith("native code-completion")
    assert audit["chat"]["chat_template"] == "tokenizer.apply_chat_template"
    assert audit["chat"]["enable_thinking"] is False
    assert audit["summarization"]["instruction"].startswith("Summarize")
    assert manifest["tokenizer_metadata"]["truncation_policy"].startswith("no truncation")
    chat = next(row for row in load_r3_workload(workload) if row.task_class == "chat")
    assert "[THINKING=False]" in chat.prompt_text
    assert "<USER_TEXT_REDACTED>" in audit["chat"]["deidentified_example"]


def test_coverage_density_missing_depth_and_horizons(phase3c2_pipeline):
    for record in labeled_store(phase3c2_pipeline["labeled"]).records():
        metrics = [
            coverage_metrics(record, set(record.pool_node_ids[ratio]), set())
            for ratio in POOL_ORDER
        ]
        recalls = [row["target_path_recall"] for row in metrics]
        assert recalls == sorted(recalls)
        for ratio, row in zip(POOL_ORDER, metrics):
            assert row["target_node_density"] == pytest.approx(
                row["target_path_nodes_present"] / len(record.pool_node_ids[ratio])
            )
            horizon = row["verification_horizon_target_recall"]
            assert set(horizon) == {"4", "8", "16"}
            first = row["first_missing_target_depth"]
            if first is not None:
                assert first >= 1
            if row["first_missing_within_verification_horizon"] is not None:
                assert row["first_missing_within_verification_horizon"] <= 4


def test_resummary_preserves_raw_artifacts_and_adds_shell_stability_stats(
    phase3c2_pipeline,
):
    raw_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (phase3c2_pipeline["labeled"], phase3c2_pipeline["selectors"])
        for path in root.rglob("*.json")
    }
    report = summarize_selector_diagnosis(
        labeled_dir=phase3c2_pipeline["labeled"],
        selector_dir=phase3c2_pipeline["selectors"],
        source_trace_commit="2" * 40,
        workload_manifest_path=phase3c2_pipeline["manifest"],
    )
    assert report["coverage_definition_version"] == COVERAGE_DEFINITION_VERSION
    assert report["coverage_semantics_audit"]["nested_target_path_recall_monotonic"]
    assert len(report["pool_shell_decomposition"]["aggregate"]) == 3
    assert report["selection_set_stability"]["aggregate"]
    assert report["headroom_decomposition"]["aggregate"]
    for row in report["aggregate_metrics"]:
        assert "request_bootstrap_95_ci" in row["statistics"]["accepted_draft_tokens_per_proposal"]
        assert "paired_delta_vs_within_request_oracle" in row
        assert "win_tie_loss_vs_residual_probability" in row
    assert raw_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_hashes
    }


def test_selection_stability_exact_match_and_jaccard(phase3c2_pipeline):
    labels = labeled_store(phase3c2_pipeline["labeled"]).records()
    rows = [
        row
        for record in selector_store(phase3c2_pipeline["selectors"]).records()
        for row in record["rows"]
    ]
    stability = _selection_stability(labels, rows)
    assert stability["per_request"]
    assert all(0 <= row["selected_set_jaccard"] <= 1 for row in stability["per_request"])
    exact = [row for row in stability["per_request"] if row["exact_match"]]
    assert all(row["selected_set_jaccard"] == 1 for row in exact)


def test_stratified_request_bootstrap_is_deterministic_and_not_node_based():
    rows = [
        {"request_id": "c1", "task_class": "code", "value": 1.0},
        {"request_id": "c2", "task_class": "code", "value": 3.0},
        {"request_id": "h1", "task_class": "chat", "value": 10.0},
    ]
    first = stratified_bootstrap_ci(rows, "value", seed=7, iterations=100)
    second = stratified_bootstrap_ci(rows, "value", seed=7, iterations=100)
    assert first == second
    assert first[0] <= sum(row["value"] for row in rows) / 3 <= first[1]


def test_multiround_common_snapshot_sequential_semantics_resume_and_headroom(
    phase3c2_pipeline, tmp_path
):
    snapshots = tmp_path / "snapshots"
    sequential = tmp_path / "sequential"
    target_bytes = {
        request.request_id: target_store(phase3c2_pipeline["target"])
        .path(request.request_id)
        .read_bytes()
        for request in phase3c2_pipeline["requests"]
    }
    snapshot_report = run_common_prefix_snapshot_stage(
        phase3c2_pipeline["requests"],
        phase3c2_pipeline["config"],
        workload_path=phase3c2_pipeline["workload"],
        target_dir=phase3c2_pipeline["target"],
        output_dir=snapshots,
        resume=False,
        backend=DryRunBackend("draft", 1664),
    )
    expected_snapshots = sum(
        target_store(phase3c2_pipeline["target"]).read(request.request_id).target_path_length
        for request in phase3c2_pipeline["requests"]
    )
    assert snapshot_report["completed_snapshots"] == expected_snapshots
    records = snapshot_store(snapshots).records()
    for record in records:
        assert record["common_snapshot_shared_by_all_selectors"]
        assert set(record["pool_sha256"]) == set(POOL_ORDER)
        assert record["remaining_target_tokens"] == len(record["target_continuation"])
    replay = run_sequential_replay_stage(
        phase3c2_pipeline["requests"],
        target_dir=phase3c2_pipeline["target"],
        snapshot_dir=snapshots,
        output_dir=sequential,
        resume=False,
    )
    assert replay["completed_records"] == len(phase3c2_pipeline["requests"])
    for record in sequential_store(sequential).records():
        assert all(row["final_sequence_matches_target"] for row in record["results"])
        assert all(row["proposal_rounds"] >= 1 for row in record["results"])
        assert all(row["verified_nodes"] >= row["accepted_tokens"] for row in record["results"])
    resumed = run_sequential_replay_stage(
        phase3c2_pipeline["requests"],
        target_dir=phase3c2_pipeline["target"],
        snapshot_dir=snapshots,
        output_dir=sequential,
        resume=True,
    )
    assert resumed["new_records"] == 0
    assert target_bytes == {
        request.request_id: target_store(phase3c2_pipeline["target"])
        .path(request.request_id)
        .read_bytes()
        for request in phase3c2_pipeline["requests"]
    }
    decomposition = multiround_headroom(records)
    assert decomposition["aggregate"]
    assert any(
        row["pool_expansion_utilization"] is None
        and row["pool_expansion_utilization_identifiable_requests"] == 0
        for row in decomposition["aggregate"]
        if row["pool_ratio"] == "1x"
    )
    summary = summarize_multiround(
        snapshot_dir=snapshots,
        sequential_dir=sequential,
        source_trace_commit="3" * 40,
    )
    assert summary["final_target_sequence_match"]
    assert not summary["gpu_performance_result"]


def test_phase3c2_cli_resummary_and_multiround_help_paths(phase3c2_pipeline, tmp_path):
    output = tmp_path / "resummary.json"
    markdown = tmp_path / "resummary.md"
    assert (
        main(
            [
                "phase3c-resummary",
                "--labeled-dir",
                str(phase3c2_pipeline["labeled"]),
                "--selector-dir",
                str(phase3c2_pipeline["selectors"]),
                "--draft-dir",
                str(phase3c2_pipeline["draft"]),
                "--target-dir",
                str(phase3c2_pipeline["target"]),
                "--source-trace-commit",
                "4" * 40,
                "--workload-manifest",
                str(phase3c2_pipeline["manifest"]),
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["source_trace_commit"] == "4" * 40
    assert set(report["source_trace_sha256"]) == {
        "draft",
        "target",
        "labeled",
        "selectors",
        "combined",
    }
    assert markdown.read_text().startswith("# Phase 3C.2")
    assert set(TARGET_BLIND_SELECTORS).issubset(report["target_blind_selectors"])

    snapshots = tmp_path / "cli-snapshots"
    sequential = tmp_path / "cli-sequential"
    assert (
        main(
            [
                "phase3c-multiround-snapshots",
                "--config",
                str(CONFIG),
                "--workload",
                str(phase3c2_pipeline["workload"]),
                "--target-dir",
                str(phase3c2_pipeline["target"]),
                "--output-dir",
                str(snapshots),
                "--backend",
                "dry-run",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "phase3c-multiround-replay",
                "--workload",
                str(phase3c2_pipeline["workload"]),
                "--target-dir",
                str(phase3c2_pipeline["target"]),
                "--snapshot-dir",
                str(snapshots),
                "--output-dir",
                str(sequential),
            ]
        )
        == 0
    )
    multi_json = tmp_path / "multi.json"
    multi_md = tmp_path / "multi.md"
    assert (
        main(
            [
                "phase3c-multiround-summary",
                "--snapshot-dir",
                str(snapshots),
                "--sequential-dir",
                str(sequential),
                "--source-trace-commit",
                "5" * 40,
                "--output",
                str(multi_json),
                "--markdown-output",
                str(multi_md),
            ]
        )
        == 0
    )
    assert json.loads(multi_json.read_text())["final_target_sequence_match"]
