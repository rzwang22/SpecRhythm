import hashlib
import json
from pathlib import Path

import pytest

from specrhythm.cli import main
from specrhythm.phase3.engine import DryRunBackend
from specrhythm.phase3.phase3c_config import (
    PublicDatasetConfig,
    load_frozen_pool_dimensions,
    load_phase3c_config,
)
from specrhythm.phase3.r3_workload import (
    TASK_SLO_MS,
    adapt_public_dataset,
    build_r3_real_workload,
    load_r3_workload,
    stable_request_split,
)
from specrhythm.phase3.real_candidate_trace import (
    TargetFeatureLeakageError,
    forest_store,
    labeled_store,
    run_draft_forest_stage,
    run_label_join_stage,
    run_target_trajectory_stage,
    target_store,
    validate_phase3c_artifacts,
)
from specrhythm.phase3.selector_diagnosis import (
    ORACLE_SELECTOR,
    SELECTOR_ORDER,
    TARGET_BLIND_SELECTORS,
    replay_request,
    run_selector_replay_stage,
    select_target_blind,
    selector_store,
    summarize_selector_diagnosis,
    validate_selector_artifacts,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = FIXTURES / "phase3c-config.json"


class CharacterTokenizer:
    model_id = "fixture:qwen-tokenizer-interface"
    tokenizer_fingerprint = "f" * 64

    def encode(self, prompt):
        return [ord(character) for character in prompt]


@pytest.fixture()
def phase3c_pipeline(tmp_path):
    config = load_phase3c_config(str(CONFIG))
    workload = tmp_path / "workload.jsonl"
    manifest = tmp_path / "workload-manifest.json"
    build_r3_real_workload(
        config,
        output_path=workload,
        manifest_path=manifest,
        command="fixture-build",
        git_commit="a" * 40,
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
    run_selector_replay_stage(
        config, labeled_dir=labeled, output_dir=selectors, resume=False
    )
    return {
        "config": config,
        "workload": workload,
        "requests": requests,
        "draft": draft,
        "target": target,
        "labeled": labeled,
        "selectors": selectors,
    }


def test_public_dataset_adapters_cover_supported_formats(tmp_path):
    cases = {
        "humaneval": ("code", {"task_id": "c", "prompt": "def f():\n    pass"}),
        "mbpp": ("code", {"task_id": "m", "text": "Write a function."}),
        "sharegpt": (
            "chat",
            {"id": "s", "conversations": [{"from": "human", "value": "Hello"}]},
        ),
        "openassistant": (
            "chat",
            {"id": "o", "messages": [{"role": "prompter", "content": "Help"}]},
        ),
        "cnn_dailymail": ("summarization", {"id": "n", "article": "News"}),
        "xsum": ("summarization", {"id": "x", "document": "Document"}),
        "govreport": ("summarization", {"id": "g", "report": "Report"}),
    }
    for adapter, (task, row) in cases.items():
        path = tmp_path / f"{adapter}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        source = PublicDatasetConfig(
            task,
            adapter,
            adapter,
            str(path),
            "test",
            f"fixture://{adapter}",
        )
        prompts = adapt_public_dataset(source)
        assert len(prompts) == 1
        assert prompts[0].prompt_text


def test_missing_dataset_fails_without_handwritten_fallback(tmp_path):
    source = PublicDatasetConfig(
        "code", "HumanEval", "humaneval", str(tmp_path / "missing.jsonl"), "test", "url"
    )
    with pytest.raises(FileNotFoundError, match="required public dataset"):
        adapt_public_dataset(source)


def test_workload_is_622_deterministic_and_uses_tokenizer_lengths(tmp_path):
    config = load_phase3c_config(str(CONFIG))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = tmp_path / "first-manifest.json"
    second_manifest = tmp_path / "second-manifest.json"
    tokenizer = CharacterTokenizer()
    one = build_r3_real_workload(
        config,
        output_path=first,
        manifest_path=first_manifest,
        command="first",
        tokenizer=tokenizer,
    )
    two = build_r3_real_workload(
        config,
        output_path=second,
        manifest_path=second_manifest,
        command="second",
        tokenizer=tokenizer,
    )
    assert first.read_bytes() == second.read_bytes()
    assert one["output_workload_sha256"] == two["output_workload_sha256"]
    assert one["task_counts"] == {"code": 3, "chat": 1, "summarization": 1}
    requests = load_r3_workload(first)
    assert [request.arrival_timestamp for request in requests] == [0, 10, 30, 60, 100]
    assert all(request.prompt_length == len(request.prompt_text) for request in requests)
    assert all(
        request.slo_class == f"{int(TASK_SLO_MS[request.task_class])}ms"
        for request in requests
    )
    assert all(
        request.data_split == stable_request_split(request.request_id)
        for request in requests
    )
    assert hashlib.sha256(first.read_bytes()).hexdigest() == one["output_workload_sha256"]


def test_nested_pools_stable_identity_and_frozen_phase2_counts(phase3c_pipeline):
    dimensions = load_frozen_pool_dimensions(phase3c_pipeline["config"])
    assert dimensions["pool_node_counts"] == {"1x": 16, "2x": 32, "4x": 64}
    records = forest_store(phase3c_pipeline["draft"]).records()
    for record in records:
        one = record.pool_node_ids["1x"]
        two = record.pool_node_ids["2x"]
        four = record.pool_node_ids["4x"]
        assert two[: len(one)] == one
        assert four[: len(two)] == two
        assert len(set(four)) == 64
        for ratio in ("1x", "2x", "4x"):
            pool = set(record.pool_node_ids[ratio])
            assert all(
                node.parent_id is None or node.parent_id in pool
                for node in record.nodes
                if node.stable_node_id in pool
            )


def test_target_trajectory_is_one_fixed_ratio_independent_record(
    phase3c_pipeline, tmp_path
):
    targets = target_store(phase3c_pipeline["target"])
    labels = labeled_store(phase3c_pipeline["labeled"])
    repeated_dir = tmp_path / "target-repeat"
    run_target_trajectory_stage(
        phase3c_pipeline["requests"],
        phase3c_pipeline["config"],
        workload_path=phase3c_pipeline["workload"],
        output_dir=repeated_dir,
        resume=False,
        backend=DryRunBackend("target", 1664),
    )
    repeated = target_store(repeated_dir)
    for request in phase3c_pipeline["requests"]:
        target = targets.read(request.request_id)
        labeled = labels.read(request.request_id)
        assert repeated.read(request.request_id) == target
        assert labeled.target_trajectory == target
        assert target.target_forward_count == target.target_path_length
        assert not target.kv_cache_reuse
        assert set(labeled.target_path_node_ids_by_pool) == {"1x", "2x", "4x"}


def test_missing_target_coverage_is_recorded_not_assumed(phase3c_pipeline):
    records = labeled_store(phase3c_pipeline["labeled"]).records()
    assert any(record.missing_target_depths_by_pool["1x"] for record in records)
    for record in records:
        assert set(record.target_path_node_ids_by_pool["1x"]).issubset(
            record.pool_node_ids["1x"]
        )


def test_target_blind_selector_rejects_labeled_nodes(phase3c_pipeline):
    record = labeled_store(phase3c_pipeline["labeled"]).records()[0]
    with pytest.raises(TargetFeatureLeakageError, match="labels are forbidden"):
        select_target_blind("residual-probability", record.nodes, 4)
    contaminated = {
        **record.nodes[0].runtime_features.__dict__,
        "on_target_path": True,
    }
    with pytest.raises(TargetFeatureLeakageError, match="forbidden fields"):
        type(record.nodes[0].runtime_features).from_dict(contaminated)


def test_all_selectors_share_forest_outcome_and_conserve_budget(phase3c_pipeline):
    labels = labeled_store(phase3c_pipeline["labeled"])
    for record in labels.records():
        replay = replay_request(record, 4)
        assert replay["selectors"] == list(SELECTOR_ORDER)
        for ratio in ("1x", "2x", "4x"):
            rows = [row for row in replay["rows"] if row["pool_ratio"] == ratio]
            assert {row["selector"] for row in rows} == set(SELECTOR_ORDER)
            assert len({row["forest_sha256"] for row in rows}) == 1
            assert len({row["target_trajectory_sha256"] for row in rows}) == 1
            for row in rows:
                accounting = row["candidate_accounting"]
                assert accounting["selected_verify_nodes"] <= 4
                assert accounting["accepted_candidate_tokens"] <= accounting[
                    "selected_verify_nodes"
                ]
                assert accounting["committed_candidate_tokens"] == accounting[
                    "accepted_candidate_tokens"
                ]
                assert accounting["committed_target_root_tokens"] in {0, 1}
                assert row["request_roots"] == 1
        assert ORACLE_SELECTOR not in TARGET_BLIND_SELECTORS


def test_target_checkpoint_resume_does_not_overwrite(phase3c_pipeline):
    store = target_store(phase3c_pipeline["target"])
    before = {
        request.request_id: store.path(request.request_id).read_bytes()
        for request in phase3c_pipeline["requests"]
    }
    report = run_target_trajectory_stage(
        phase3c_pipeline["requests"],
        phase3c_pipeline["config"],
        workload_path=phase3c_pipeline["workload"],
        output_dir=phase3c_pipeline["target"],
        resume=True,
        backend=DryRunBackend("target", 1664),
    )
    assert report["new_records"] == 0
    assert before == {
        request.request_id: store.path(request.request_id).read_bytes()
        for request in phase3c_pipeline["requests"]
    }


def test_validation_and_selector_diagnostics_are_non_performance(phase3c_pipeline):
    validation = validate_phase3c_artifacts(
        phase3c_pipeline["requests"],
        forest_dir=phase3c_pipeline["draft"],
        target_dir=phase3c_pipeline["target"],
        labeled_dir=phase3c_pipeline["labeled"],
    )
    assert validation["valid"]
    report = summarize_selector_diagnosis(
        labeled_dir=phase3c_pipeline["labeled"],
        selector_dir=phase3c_pipeline["selectors"],
    )
    assert not report["gpu_performance_result"]
    assert not report["reports_goodput"]
    assert not report["reports_slo_attainment"]
    assert not report["reports_speedup"]
    assert report["calibration"]["depth_probability_bins"]
    assert report["calibration"]["depth_entropy_bins"]
    assert report["calibration"]["sibling_rank_target_hit_rate"]
    assert report["calibration"]["selected_vs_unselected"]
    assert report["pool_expansion_robustness"]


def test_selector_validation_rejects_budget_accounting_corruption(phase3c_pipeline):
    store = selector_store(phase3c_pipeline["selectors"])
    request_id = phase3c_pipeline["requests"][0].request_id
    path = store.path(request_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["rows"][0]["candidate_accounting"]["selected_verify_nodes"] = 99
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    validation = validate_selector_artifacts(
        labeled_dir=phase3c_pipeline["labeled"],
        selector_dir=phase3c_pipeline["selectors"],
    )
    assert not validation["valid"]
    assert any("verified-node accounting" in error for error in validation["errors"])


def test_phase3c_cli_dry_run_and_resume(tmp_path):
    workload = tmp_path / "workload.jsonl"
    manifest = tmp_path / "manifest.json"
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    labeled = tmp_path / "labeled"
    selectors = tmp_path / "selectors"
    assert main(
        [
            "phase3c-workload-build",
            "--config",
            str(CONFIG),
            "--output",
            str(workload),
            "--manifest",
            str(manifest),
            "--backend",
            "dry-run",
        ]
    ) == 0
    for command, output in (
        ("phase3c-draft-forest", draft),
        ("phase3c-target-trajectory", target),
    ):
        assert main(
            [
                command,
                "--config",
                str(CONFIG),
                "--workload",
                str(workload),
                "--output-dir",
                str(output),
                "--backend",
                "dry-run",
            ]
        ) == 0
    assert main(
        [
            "phase3c-label-join",
            "--workload",
            str(workload),
            "--forest-dir",
            str(draft),
            "--target-dir",
            str(target),
            "--output-dir",
            str(labeled),
        ]
    ) == 0
    assert main(
        [
            "phase3c-selector-replay",
            "--config",
            str(CONFIG),
            "--labeled-dir",
            str(labeled),
            "--output-dir",
            str(selectors),
        ]
    ) == 0
    validation = tmp_path / "validation.json"
    assert main(
        [
            "phase3c-validate",
            "--workload",
            str(workload),
            "--forest-dir",
            str(draft),
            "--target-dir",
            str(target),
            "--labeled-dir",
            str(labeled),
            "--selector-dir",
            str(selectors),
            "--output",
            str(validation),
        ]
    ) == 0
    assert json.loads(validation.read_text())["valid"]
    assert len(selector_store(selectors).records()) == 5
