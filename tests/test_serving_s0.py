"""Synthetic contract tests only; no downloaded data, real tokenizer, model or GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrhythm.serving.arrival import arrival_summary, read_arrivals, scheduled_arrival, windows
from specrhythm.serving.builder import build, output_names, plan
from specrhythm.serving.cli import main
from specrhythm.serving.common import (
    LOCK_SCHEMA,
    TASKS,
    DataError,
    canonical,
    digest,
    read_json,
    sha256_file,
    write_json,
)
from specrhythm.serving.schema import (
    ServingWorkloadRequest,
    load_requests,
    require_calibrated,
)
from specrhythm.serving.selection import choose, extract
from specrhythm.serving.sources import DATASETS, records, validate_lock, verify_sources
from specrhythm.serving.tokenization import QwenTokenizer, check_tokenizers
from specrhythm.serving.validation import rebuild_check, seal, validate

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/serving-s0"


class FixtureTokenizer:
    is_real = False
    fingerprint = digest("synthetic-character-tokenizer-v1; NOT Qwen3")
    metadata = {"kind": "synthetic-fixture-tokenizer", "fingerprint": fingerprint}

    def render(self, messages):
        return "<fixture-user>" + messages[0]["content"] + "</fixture-user><fixture-assistant>"

    def encode(self, text):
        return [ord(c) for c in text]


@pytest.fixture
def inputs(tmp_path):
    sources = tmp_path / "sources"
    lock = {"schema_version": LOCK_SCHEMA, "data_kind": "synthetic-fixture", "sources": {}}
    for task in (*TASKS, "arrival", "tokenizer"):
        directory = sources / task
        directory.mkdir(parents=True)
        name = task + ".jsonl" if task != "tokenizer" else "fixture-tokenizer.txt"
        if task == "tokenizer":
            (directory / name).write_text("synthetic character tokenizer; not a real tokenizer")
        else:
            shutil.copyfile(FIXTURES / name, directory / name)
        file = {
            "path": name,
            "sha256": sha256_file(directory / name),
            "role": "data" if task != "tokenizer" else "tokenizer",
        }
        source = {"revision": "a" * 40, "files": [file], "license_declared": "synthetic-test-only"}
        if task in DATASETS:
            source.update(dataset_id=DATASETS[task][0], config=DATASETS[task][1], split="train")
        else:
            source["repository_id"] = "synthetic-fixture/" + task
        lock["sources"][task] = source
    lock_path = tmp_path / "source-lock.json"
    write_json(lock_path, lock)
    config = read_json(REPO / "configs/workloads/mixed-real-v1.json")
    config["workload_family"] = "synthetic-s0-fixture"
    for split, quota in (("main", 2), ("calibration", 1)):
        config["splits"][split] = {
            "workload_id": f"synthetic-{split}",
            "quotas": {t: quota for t in TASKS},
        }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    return SimpleNamespace(
        root=tmp_path,
        sources=sources,
        lock=lock,
        lock_path=lock_path,
        config=config,
        config_path=config_path,
        tokenizer=FixtureTokenizer(),
    )


def construct(inputs, name="build-a"):
    output = inputs.root / name
    report = build(
        inputs.config_path,
        inputs.lock_path,
        inputs.sources,
        inputs.tokenizer,
        output,
        tokenizer_path=Path("synthetic-fixture-tokenizer"),
        allow_synthetic=True,
    )
    assert report["valid"], report
    return output


def mutate_source(inputs, task, transform):
    path = inputs.sources / task / (task + ".jsonl")
    data = [json.loads(r) for r in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in transform(data)))
    inputs.lock["sources"][task]["files"][0]["sha256"] = sha256_file(path)
    inputs.lock_path.write_bytes(canonical(inputs.lock))


def test_build_full_independent_validation_and_cross_directory_determinism(inputs):
    a = construct(inputs)
    b = construct(inputs, "another/location/build-b")
    assert rebuild_check(a, b)["valid"]
    names = output_names(inputs.config)
    rows = {s: load_requests(a / n) for s, n in names.items()}
    assert {s: len(r) for s, r in rows.items()} == {"main": 8, "calibration": 4}
    combined = [r for rs in rows.values() for r in rs]
    assert len({r.request_id for r in combined}) == 12
    assert all(r.arrival_time_ms >= 0 for r in combined)
    assert read_json(a / "validation.json")["code_fixture_validation"] == "PASS"
    assert read_json(a / "validation.json")["real_data_validation"] == "PENDING"
    assert read_json(a / "validation.json")["server_local_tokenizer_verification"] == "PENDING"
    assert not any(
        word in (a / "sample-review.md").read_text() for word in ("SECRET-", "NOT-A-REAL-PREFIX")
    )
    assert (
        a.joinpath("workload-manifest.json").read_bytes()
        == b.joinpath("workload-manifest.json").read_bytes()
    )
    assert read_json(a / "build-info.json") != read_json(b / "build-info.json")


def test_loader_supports_arbitrary_count_and_rejects_trailing_row(inputs):
    root = construct(inputs)
    path = root / "main8.jsonl"
    original = path.read_bytes()
    path.write_bytes(original.splitlines(keepends=True)[0])
    assert len(load_requests(path)) == 1
    path.write_bytes(original + original.splitlines(keepends=True)[0])
    report = validate(root, inputs.sources, inputs.tokenizer, allow_synthetic=True)
    assert not report["valid"]
    assert "duplicate serving request ID" in report["errors"][0]


@pytest.mark.parametrize(
    "task, secret",
    [
        ("code", "SECRET-SOLUTION"),
        ("summarization", "SECRET-REFERENCE-SUMMARY"),
        ("reasoning", "SECRET-REFERENCE-REASONING"),
    ],
)
def test_answer_isolation_in_source_adapters(inputs, task, secret):
    row = json.loads((FIXTURES / (task + ".jsonl")).read_text().splitlines()[0])
    prompt, reason = extract(task, row, inputs.config)
    assert reason is None and secret not in prompt["content"]
    if task == "code":
        assert "def solve():\n    pass\n" in prompt["content"]
        assert "SECRET-HIDDEN-TESTS" not in prompt["content"]


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("role", "assistant", "chat_non_prompter"),
        ("lang", "de", "chat_non_english"),
        ("parent_id", "parent", "chat_non_root"),
        ("deleted", True, "chat_deleted"),
        ("text", "Write Python code", "chat_code_task"),
        ("text", "Solve this equation", "chat_math_task"),
        ("text", "The value of X is 4; Y is 3*x and Z is X+Y. What is Z?", "chat_math_task"),
        ("text", "Summarize this article", "chat_summary_task"),
    ],
)
def test_chat_fixed_filter_rules(inputs, field, value, reason):
    row = json.loads((FIXTURES / "chat.jsonl").read_text().splitlines()[0])
    row[field] = value
    assert extract("chat", row, inputs.config) == (None, reason)


def test_joint_dedup_and_refill(inputs):
    def change(rows):
        same_id = deepcopy(rows[0])
        same_group = {**rows[1], "message_id": "another-id"}
        same_prompt = {**rows[2], "message_id": "another-root", "message_tree_id": "another-group"}
        return rows + [same_id, same_group, same_prompt]

    mutate_source(inputs, "chat", change)
    root = construct(inputs)
    stats = read_json(root / "selection-report.json")["by_class"]["chat"]
    assert sum(stats["duplicates"].values()) == 3
    assert stats["selected"] == {"main": 2, "calibration": 1}


def test_lengths_are_filtered_without_truncation_and_refilled(inputs):
    mutate_source(
        inputs,
        "summarization",
        lambda rows: [*rows, {"id": "too-long", "article": "x" * 4000, "highlights": "SECRET"}],
    )
    # Ask for every short article so the long one must be examined.
    inputs.config["splits"]["main"]["quotas"]["summarization"] = 6
    inputs.config_path.write_bytes(canonical(inputs.config))
    chosen, report = choose(inputs.lock, inputs.sources, inputs.config, inputs.tokenizer)
    assert sum(len(chosen[s]["summarization"]) for s in chosen) == 7
    # Its hash rank may follow all seven valid candidates; only examined lengths are reported.
    assert report["by_class"]["summarization"]["length_checked"] >= 7
    assert all(
        "x" * 4000 not in r["prompt_text"] for s in chosen for r in chosen[s]["summarization"]
    )


def test_candidate_exhaustion_is_explicit_and_never_duplicates(inputs):
    mutate_source(inputs, "reasoning", lambda rows: rows[:2])
    with pytest.raises(DataError, match="candidate pool exhausted"):
        construct(inputs)
    failure = read_json(inputs.root / "build-a/failure-report.json")
    assert failure["details"]["actual"] < failure["details"]["required"]
    assert "selection_report" in failure["details"]


def test_missing_source_field_and_checksum_fail_closed(inputs):
    path = inputs.sources / "code/code.jsonl"
    path.write_text("{}\n")
    with pytest.raises(DataError, match="checksum"):
        verify_sources(inputs.lock, inputs.sources, allow_synthetic=True)
    with pytest.raises(DataError, match="fields missing"):
        list(records(path, ["id", "question"]))


def test_arrival_ties_original_lines_windows_and_scale(inputs):
    rows = read_arrivals(inputs.lock, inputs.sources)
    assert [(r["source_timestamp_ms"], r["original_line_index"]) for r in rows[:4]] == [
        (0, 2),
        (5, 3),
        (5, 4),
        (10, 5),
    ]
    selected = windows(rows, inputs.config)
    assert selected["calibration"][0]["sorted_index"] == 8
    assert selected["main"][0]["selected_first_timestamp_ms"] == 0
    assert selected["calibration"][0]["selected_first_timestamp_ms"] == 35
    assert scheduled_arrival(10, 2) == 5
    with pytest.raises(DataError):
        scheduled_arrival(10, 0)
    with pytest.raises(DataError, match="overflows"):
        scheduled_arrival(1e300, 1e-300)
    assert (
        arrival_summary([SimpleNamespace(arrival_time_ms=0)])["observed_iat_rate_per_second"]
        is None
    )
    assert (
        arrival_summary([SimpleNamespace(arrival_time_ms=0)] * 2)["observed_iat_rate_per_second"]
        is None
    )


@pytest.mark.parametrize("value", [True, -1, float("inf"), None, "3", 10**500])
def test_invalid_original_arrival_timestamp_rejected(inputs, value):
    mutate_source(inputs, "arrival", lambda rows: [{**rows[0], "timestamp": value}, *rows[1:]])
    with pytest.raises(DataError, match="invalid Mooncake timestamp"):
        read_arrivals(inputs.lock, inputs.sources)


@pytest.mark.parametrize(
    "mutation",
    [
        "tokens",
        "arrival",
        "quota",
        "provenance",
        "messages",
        "budget",
        "seed",
        "source_group",
        "manifest",
        "summary",
        "count",
    ],
)
def test_independent_validator_rejects_tampering_and_never_mutates_inputs(inputs, mutation):
    root = construct(inputs)
    path = root / "main8.jsonl"
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    if mutation == "tokens":
        rows[0]["prompt_token_ids"][0] += 1
    elif mutation == "arrival":
        rows[0]["arrival_time_ms"] = 99
    elif mutation == "quota":
        rows[0]["task_class"] = rows[0]["slo_class"] = (
            "code" if rows[0]["task_class"] != "code" else "chat"
        )
    elif mutation == "provenance":
        rows[0]["source_ref"]["revision"] = "b" * 40
    elif mutation == "messages":
        rows[0]["messages"][0]["content"] += "INJECTED-ANSWER"
    elif mutation == "budget":
        rows[0]["maximum_new_tokens"] = 4096
    elif mutation == "seed":
        rows[0]["sampling_seed"] += 1
    elif mutation == "source_group":
        rows[1]["source_ref"]["dataset_id"] = rows[0]["source_ref"]["dataset_id"]
        rows[1]["source_ref"]["group_id"] = rows[0]["source_ref"]["group_id"]
    elif mutation == "count":
        rows.pop()
    elif mutation == "manifest":
        value = read_json(root / "workload-manifest.json")
        value["seed"] = 999
        (root / "workload-manifest.json").write_bytes(canonical(value))
    else:
        (root / "dataset-summary.json").write_text("{}\n")
    path.write_bytes(b"".join(canonical(r) for r in rows))
    before = {
        p: sha256_file(p)
        for p in [*root.iterdir(), *inputs.sources.rglob("*.jsonl")]
        if p.is_file()
    }
    report = validate(root, inputs.sources, inputs.tokenizer, allow_synthetic=True)
    assert not report["valid"], mutation
    assert before == {p: sha256_file(p) for p in before}
    assert report["inputs_unchanged"]


def test_real_mode_rejects_fixtures_and_pending_slo(inputs):
    with pytest.raises(DataError, match="fixture mode"):
        validate_lock(inputs.lock)
    with pytest.raises(DataError, match="fixture tokenizer"):
        plan(
            inputs.config,
            {**inputs.lock, "data_kind": "real-public"},
            inputs.sources,
            inputs.tokenizer,
        )
    with pytest.raises(DataError, match="pending"):
        require_calibrated(inputs.config)


def test_request_identity_survives_arrival_configuration_change(inputs):
    original, _, _ = plan(inputs.config, inputs.lock, inputs.sources, inputs.tokenizer)
    config = deepcopy(inputs.config)
    config["arrival_seed"] += 1
    changed, _, _ = plan(config, inputs.lock, inputs.sources, inputs.tokenizer)

    def identity(rows):
        return {r.request_id: r.sampling_seed for data in rows.values() for r in data}

    assert identity(original) == identity(changed)
    assert [r.request_id for r in original["main"]] != [r.request_id for r in changed["main"]]


def test_cli_help_imports_without_optional_dependencies():
    command = subprocess.run(
        [sys.executable, "-m", "specrhythm.serving", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert command.returncode == 0
    for name in (
        "source-resolve",
        "source-fetch",
        "build",
        "validate",
        "inspect",
        "rebuild-check",
    ):
        assert name in command.stdout


def test_cli_explicit_error_and_no_fake_root_creation(tmp_path, capsys):
    assert (
        main(
            [
                "build",
                "--sources-dir",
                str(tmp_path),
                "--tokenizer-path",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "new"),
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert not report["valid"] and "tokenizer directory missing" in report["errors"][0]
    assert not (tmp_path / "new").exists()


def test_seal_checksums_exclude_self_and_bundle_raw_sources(inputs):
    root = construct(inputs)
    bundle = inputs.root / "review.tar.gz"
    result = seal(root, bundle=bundle)
    assert "checksums.sha256" not in result["artifacts"]
    assert {"stdout.log", "stderr.log", "build-info.json"} <= set(result["artifacts"])
    import tarfile

    with tarfile.open(bundle) as archive:
        assert all("/sources/" not in name for name in archive.getnames())
    with pytest.raises(DataError, match="fresh"):
        seal(root)


def test_qwen_template_flags_and_no_duplicate_special_tokens(tmp_path, monkeypatch):
    calls = []
    fake = SimpleNamespace(
        is_fast=True,
        chat_template="enable_thinking template",
        special_tokens_map={"eos_token": "EOS"},
        all_special_ids=[1],
        get_vocab=lambda: {"EOS": 1, "hi": 2},
        backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"),
        apply_chat_template=lambda messages, **kwargs: calls.append(kwargs) or "EOS hi",
        encode=lambda text, **kwargs: calls.append(kwargs) or [1, 2],
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="fixture",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: fake),
        ),
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    tokenizer = QwenTokenizer(tmp_path)
    assert tokenizer.encode(tokenizer.render([{"role": "user", "content": "hi"}])) == [1, 2]
    assert calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        {"add_special_tokens": False},
    ]


def test_server_tokenizer_mismatch_fails_closed():
    a = SimpleNamespace(is_real=True, identity={"EOS": 1})
    b = SimpleNamespace(is_real=True, identity={"EOS": 2})
    with pytest.raises(DataError, match="mapping differs"):
        check_tokenizers(a, b, [])


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompt_length", True),
        ("maximum_new_tokens", True),
        ("sampling_seed", False),
        ("arrival_time_ms", float("nan")),
        ("language", "German"),
        ("schema_version", "unknown"),
        ("prompt_token_ids", [True]),
    ],
)
def test_schema_rejects_invalid_types(inputs, field, value):
    rows, _, _ = plan(inputs.config, inputs.lock, inputs.sources, inputs.tokenizer)
    raw = rows["main"][0].to_dict()
    raw[field] = value
    with pytest.raises((DataError, ValueError)):
        ServingWorkloadRequest.from_dict(raw)


@pytest.mark.parametrize(
    "field,value", [("id", 7), ("group_id", []), ("original_row_index", True)]
)
def test_loader_rejects_malformed_source_identity(inputs, field, value):
    rows, _, _ = plan(inputs.config, inputs.lock, inputs.sources, inputs.tokenizer)
    raw = rows["main"][0].to_dict()
    raw["source_ref"][field] = value
    with pytest.raises(DataError, match="dataset source reference"):
        ServingWorkloadRequest.from_dict(raw)


def test_extra_unique_row_is_not_hidden_by_count_limit(inputs):
    root = construct(inputs)
    main_path = root / "main8.jsonl"
    extra = (root / "calibration4.jsonl").read_bytes().splitlines(keepends=True)[0]
    main_path.write_bytes(main_path.read_bytes() + extra)
    assert len(load_requests(main_path)) == 9
    result = validate(root, inputs.sources, inputs.tokenizer, allow_synthetic=True)
    assert not result["valid"] and "full-file request count differs" in result["errors"][0]


def test_validator_checks_trusted_config_and_lock_and_source_changes(inputs):
    root = construct(inputs)
    path = root / "source-lock.json"
    original = path.read_bytes()
    lock = read_json(path)
    lock["sources"]["chat"]["license_declared"] = "tampered"
    path.write_bytes(canonical(lock))
    report = validate(
        root,
        inputs.sources,
        inputs.tokenizer,
        allow_synthetic=True,
        expected_lock=inputs.lock_path,
    )
    assert report["errors"] == ["build source lock differs from trusted lock"]
    path.write_bytes(original)
    config_path = root / "workload-config.json"
    config = read_json(config_path)
    config["seed"] += 1
    config_path.write_bytes(canonical(config))
    report = validate(
        root,
        inputs.sources,
        inputs.tokenizer,
        allow_synthetic=True,
        expected_config=inputs.config_path,
    )
    assert report["errors"] == ["build config differs from expected config"]
    config_path.write_bytes(inputs.config_path.read_bytes())
    source = inputs.sources / "code/code.jsonl"
    source.write_bytes(source.read_bytes() + b"{}\n")
    report = validate(root, inputs.sources, inputs.tokenizer, allow_synthetic=True)
    assert report["errors"] == ["source checksum mismatch"]


def test_cross_split_duplicate_and_post_validation_change_cannot_pass(inputs):
    root = construct(inputs)
    a, b = load_requests(root / "main8.jsonl"), load_requests(root / "calibration4.jsonl")
    replacement = next(r for r in a if r.task_class == b[0].task_class).to_dict()
    replacement.update(split="calibration", workload_id="synthetic-calibration")
    path = root / "calibration4.jsonl"
    path.write_bytes(canonical(replacement) + b"".join(canonical(r.to_dict()) for r in b[1:]))
    report = validate(root, inputs.sources, inputs.tokenizer, allow_synthetic=True)
    assert not report["valid"]
    with pytest.raises(DataError, match="changed after data validation"):
        seal(root)


def test_context_reserve_is_enforced_without_truncation(inputs):
    config = deepcopy(inputs.config)
    config["max_model_len"] = config["class_limits"]["reasoning"]["maximum_new_tokens"] + 4
    with pytest.raises(DataError, match="exhausted") as error:
        choose(inputs.lock, inputs.sources, config, inputs.tokenizer)
    counts = error.value.details["selection_report"]["by_class"]["reasoning"]
    assert counts["filters"]["context_budget"] == 7
    assert counts["selected"] == {"main": 0, "calibration": 0}


def test_trace_validates_unselected_tail_and_reports_short_pool(inputs):
    mutate_source(inputs, "arrival", lambda rows: [*rows[:-1], {**rows[-1], "timestamp": -1}])
    with pytest.raises(DataError, match="timestamp"):
        read_arrivals(inputs.lock, inputs.sources)
    with pytest.raises(DataError, match="insufficient original Mooncake arrivals"):
        windows([], inputs.config)


def test_line_ending_dedup_preserves_code_indentation():
    from specrhythm.serving.common import normalized_prompt

    assert normalized_prompt("def f():\r\n    return 1\r\n") == "def f():\n    return 1\n"
    assert normalized_prompt("def f():\n    return 1\n") != normalized_prompt(
        "def f():\n return 1\n"
    )


def test_local_import_verifies_hash_without_network(inputs, tmp_path, monkeypatch):
    import specrhythm.serving.sources as module

    # Exercise transport using explicitly synthetic files. Production CLI rejects fixture locks.
    original_validate, original_verify = module.validate_lock, module.verify_sources
    monkeypatch.setattr(
        module, "validate_lock", lambda lock, **kw: original_validate(lock, allow_synthetic=True)
    )
    monkeypatch.setattr(
        module,
        "verify_sources",
        lambda lock, root, **kw: original_verify(lock, root, allow_synthetic=True),
    )
    monkeypatch.setattr(module, "_hub_file", lambda *a, **kw: pytest.fail("unexpected network"))
    monkeypatch.setattr(module, "_http_file", lambda *a, **kw: pytest.fail("unexpected network"))
    imports = {
        f"{group}/{file['path']}": str(inputs.sources / group / file["path"])
        for group, source in inputs.lock["sources"].items()
        for file in source["files"]
    }
    mapping = tmp_path / "imports.json"
    write_json(mapping, imports)
    destination = tmp_path / "imported"
    hashes = module.fetch(
        inputs.lock_path, destination, tmp_path / "cache", import_map=mapping, offline=True
    )
    assert len(hashes) == 6
    assert (
        module.fetch(
            inputs.lock_path, destination, tmp_path / "cache", import_map=mapping, offline=True
        )
        == hashes
    )
    path = inputs.sources / "chat/chat.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(DataError, match="checksum mismatch"):
        module.fetch(inputs.lock_path, destination, tmp_path / "cache", import_map=mapping)


def test_fixed_revisions_and_full_source_file_mapping_are_required():
    lock = read_json(REPO / "configs/workloads/mixed-real-v1-source-lock.json")
    validate_lock(lock)
    changed = deepcopy(lock)
    changed["sources"]["chat"]["revision"] = "main"
    with pytest.raises(DataError, match="exact commit"):
        validate_lock(changed)
    changed = deepcopy(lock)
    changed["sources"]["summarization"]["files"] = changed["sources"]["summarization"]["files"][1:]
    with pytest.raises(DataError, match="full train"):
        validate_lock(changed)
    assert all("safetensors" not in f["path"] for f in lock["sources"]["tokenizer"]["files"])


def test_real_tokenizer_must_match_locked_source(inputs, monkeypatch):
    import specrhythm.serving.tokenization as module

    lock = {**inputs.lock, "data_kind": "real-public"}
    monkeypatch.setattr(module, "QwenTokenizer", lambda p: SimpleNamespace(fingerprint="locked"))
    with pytest.raises(DataError, match="fixed-revision source lock"):
        module.verify_locked_tokenizer(
            SimpleNamespace(is_real=True, fingerprint="other"), lock, inputs.sources
        )


def test_server_tokenizer_encoding_mismatch_and_positive_coverage(inputs):
    rows, _, _ = plan(inputs.config, inputs.lock, inputs.sources, inputs.tokenizer)
    requests = [r for group in rows.values() for r in group]

    class RealInterfaceFixture(FixtureTokenizer):
        is_real = True  # Mock of the real-tokenizer interface, never used for real data.
        identity = {"fixture": 1}

    left, right = RealInterfaceFixture(), RealInterfaceFixture()
    assert check_tokenizers(left, right, requests)["validated_prompt_count"] == 12
    right.encode = lambda text: [0]
    with pytest.raises(DataError, match="encoding differs"):
        check_tokenizers(left, right, requests)
