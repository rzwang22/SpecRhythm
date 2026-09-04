from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "integrations" / "vllm" / "manage_patch.py"
)
SPEC = importlib.util.spec_from_file_location("manage_vllm_patch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


def _actual(runner: str, scheduler: str):
    return {
        str(manager.TARGET_FILE): runner,
        str(manager.SCHEDULER_FILE): scheduler,
    }


def _report(actual, expected_state):
    return manager.build_check_report(
        root=Path("/installed/vllm"),
        actual=actual,
        expected_state=expected_state,
        verified_source_commit=manager.BASE_COMMIT,
    )


def test_stock_check_accepts_only_exact_stock_state():
    stock = _actual(manager.BASE_SHA256, manager.SCHEDULER_BASE_SHA256)
    patched = _actual(manager.PATCHED_SHA256, manager.SCHEDULER_PATCHED_SHA256)
    assert _report(stock, "stock")["valid"] is True
    assert _report(patched, "stock")["valid"] is False


def test_patched_check_accepts_only_exact_patched_state():
    stock = _actual(manager.BASE_SHA256, manager.SCHEDULER_BASE_SHA256)
    patched = _actual(manager.PATCHED_SHA256, manager.SCHEDULER_PATCHED_SHA256)
    report = _report(patched, "patched")
    assert report["valid"] is True
    assert report["actual_runner_sha256"] == manager.PATCHED_SHA256
    assert report["actual_scheduler_sha256"] == manager.SCHEDULER_PATCHED_SHA256
    assert report["pinned_source_commit"] == manager.BASE_COMMIT
    assert len(report["active_patch_hashes"]) == 4
    assert _report(stock, "patched")["valid"] is False
    old_observer = _actual(
        manager.PRE_GENERIC_NUMERICAL_PATCHED_SHA256,
        manager.SCHEDULER_PATCHED_SHA256,
    )
    assert _report(old_observer, "patched")["valid"] is False


@pytest.mark.parametrize(
    ("runner", "scheduler"),
    (
        (manager.PATCHED_SHA256, manager.SCHEDULER_BASE_SHA256),
        (manager.BASE_SHA256, manager.SCHEDULER_PATCHED_SHA256),
        (manager.WORKER_HOOKS_SHA256, manager.SCHEDULER_PATCHED_SHA256),
    ),
)
def test_patched_check_rejects_partial_patch_combinations(runner, scheduler):
    report = _report(_actual(runner, scheduler), "patched")
    assert report["valid"] is False
    assert report["errors"]


def test_failed_check_writes_immutable_diagnostic_manifest(tmp_path):
    manifest = tmp_path / "check.json"
    actual = _actual(manager.PATCHED_SHA256, manager.SCHEDULER_BASE_SHA256)
    with pytest.raises(SystemExit, match="refusing check --expect-state patched"):
        manager.run_state_check(
            root=tmp_path,
            actual=actual,
            expected_state="patched",
            verified_source_commit=manager.BASE_COMMIT,
            manifest=manifest,
        )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["expected_state"] == "patched"
    assert value["valid"] is False
    with pytest.raises(FileExistsError):
        manager.run_state_check(
            root=tmp_path,
            actual=_actual(
                manager.PATCHED_SHA256, manager.SCHEDULER_PATCHED_SHA256
            ),
            expected_state="patched",
            verified_source_commit=manager.BASE_COMMIT,
            manifest=manifest,
        )


def test_apply_then_e73_patched_check_and_restore_then_stock_check(
    tmp_path, monkeypatch
):
    assert manager.PATCHED_SHA256 == (
        "a8b56ee511ad04d4f6e56e802417e6b8fb8b723a9fef05de36148f4218e9e945"
    )
    root = tmp_path / "site-packages"
    target = root / manager.TARGET_FILE
    scheduler = root / manager.SCHEDULER_FILE
    target.parent.mkdir(parents=True)
    scheduler.parent.mkdir(parents=True)
    target.write_text("runner\n", encoding="utf-8")
    scheduler.write_text("scheduler\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    state = {
        str(manager.TARGET_FILE): manager.BASE_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_BASE_SHA256,
    }

    def fake_sha256(path):
        if path == target:
            return state[str(manager.TARGET_FILE)]
        if path == scheduler:
            return state[str(manager.SCHEDULER_FILE)]
        return path.name.ljust(64, "0")[:64]

    def fake_patch(_root, *, patch, reverse, dry_run):
        if not dry_run:
            if patch == manager.PATCH:
                state[str(manager.TARGET_FILE)] = (
                    manager.BASE_SHA256 if reverse else manager.WORKER_HOOKS_SHA256
                )
            elif patch == manager.SCHEDULER_PATCH:
                state[str(manager.SCHEDULER_FILE)] = (
                    manager.SCHEDULER_BASE_SHA256
                    if reverse
                    else manager.SCHEDULER_PATCHED_SHA256
                )
            elif patch == manager.TIMING_PATCH:
                state[str(manager.TARGET_FILE)] = (
                    manager.WORKER_HOOKS_SHA256
                    if reverse
                    else manager.TIMING_PATCHED_SHA256
                )
            elif patch == manager.NUMERICAL_PATCH:
                state[str(manager.TARGET_FILE)] = (
                    manager.TIMING_PATCHED_SHA256
                    if reverse
                    else manager.PATCHED_SHA256
                )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(manager, "sha256", fake_sha256)
    monkeypatch.setattr(manager, "run_patch", fake_patch)
    monkeypatch.setattr(manager, "source_commit", lambda _source: manager.BASE_COMMIT)

    assert manager.main(["apply", "--vllm-root", str(root), "--source", str(source)]) == 0
    patched_manifest = tmp_path / "patched-check.json"
    assert (
        manager.main(
            [
                "check",
                "--vllm-root",
                str(root),
                "--source",
                str(source),
                "--expect-state",
                "patched",
                "--manifest",
                str(patched_manifest),
            ]
        )
        == 0
    )
    assert json.loads(patched_manifest.read_text())["valid"] is True

    assert (
        manager.main(["restore", "--vllm-root", str(root), "--source", str(source)])
        == 0
    )
    stock_manifest = tmp_path / "stock-check.json"
    assert (
        manager.main(
            [
                "check",
                "--vllm-root",
                str(root),
                "--source",
                str(source),
                "--expect-state",
                "stock",
                "--manifest",
                str(stock_manifest),
            ]
        )
        == 0
    )
    assert json.loads(stock_manifest.read_text())["valid"] is True


def test_restore_accepts_exact_pre_gate3_patch_state(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    target = root / manager.TARGET_FILE
    scheduler = root / manager.SCHEDULER_FILE
    target.parent.mkdir(parents=True)
    scheduler.parent.mkdir(parents=True)
    target.write_text("runner\n", encoding="utf-8")
    scheduler.write_text("scheduler\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    state = {
        str(manager.TARGET_FILE): manager.PRE_GATE3_PATCHED_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_PATCHED_SHA256,
    }
    applied = []

    def fake_sha256(path):
        if path == target:
            return state[str(manager.TARGET_FILE)]
        if path == scheduler:
            return state[str(manager.SCHEDULER_FILE)]
        return path.name.ljust(64, "0")[:64]

    def fake_patch(_root, *, patch, reverse, dry_run):
        if not dry_run:
            applied.append((patch, reverse))
            if patch == manager.TIMING_PATCH and reverse:
                state[str(manager.TARGET_FILE)] = (
                    manager.PRE_GATE3_WORKER_HOOKS_SHA256
                )
            elif patch == manager.SCHEDULER_PATCH and reverse:
                state[str(manager.SCHEDULER_FILE)] = manager.SCHEDULER_BASE_SHA256
            elif patch == manager.PRE_GATE3_PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.BASE_SHA256
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(manager, "sha256", fake_sha256)
    monkeypatch.setattr(manager, "run_patch", fake_patch)
    monkeypatch.setattr(manager, "source_commit", lambda _source: manager.BASE_COMMIT)

    assert manager.main(["restore", "--vllm-root", str(root), "--source", str(source)]) == 0
    assert (manager.PRE_GATE3_PATCH, True) in applied
    assert (manager.PATCH, True) not in applied
    assert state == {
        str(manager.TARGET_FILE): manager.BASE_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_BASE_SHA256,
    }


def test_restore_accepts_exact_pre_numerical_observer_state(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    target = root / manager.TARGET_FILE
    scheduler = root / manager.SCHEDULER_FILE
    target.parent.mkdir(parents=True)
    scheduler.parent.mkdir(parents=True)
    target.write_text("runner\n", encoding="utf-8")
    scheduler.write_text("scheduler\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    state = {
        str(manager.TARGET_FILE): manager.TIMING_PATCHED_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_PATCHED_SHA256,
    }
    applied = []

    def fake_sha256(path):
        if path == target:
            return state[str(manager.TARGET_FILE)]
        if path == scheduler:
            return state[str(manager.SCHEDULER_FILE)]
        return path.name.ljust(64, "0")[:64]

    def fake_patch(_root, *, patch, reverse, dry_run):
        if not dry_run:
            applied.append((patch, reverse))
            if patch == manager.TIMING_PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.WORKER_HOOKS_SHA256
            elif patch == manager.SCHEDULER_PATCH and reverse:
                state[str(manager.SCHEDULER_FILE)] = manager.SCHEDULER_BASE_SHA256
            elif patch == manager.PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.BASE_SHA256
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(manager, "sha256", fake_sha256)
    monkeypatch.setattr(manager, "run_patch", fake_patch)
    monkeypatch.setattr(manager, "source_commit", lambda _source: manager.BASE_COMMIT)

    assert manager.main(["restore", "--vllm-root", str(root), "--source", str(source)]) == 0
    assert (manager.NUMERICAL_PATCH, True) not in applied
    assert (manager.TIMING_PATCH, True) in applied
    assert state == {
        str(manager.TARGET_FILE): manager.BASE_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_BASE_SHA256,
    }


def test_restore_accepts_exact_c142_pre_generic_observer_state(
    tmp_path, monkeypatch
):
    root = tmp_path / "site-packages"
    target = root / manager.TARGET_FILE
    scheduler = root / manager.SCHEDULER_FILE
    target.parent.mkdir(parents=True)
    scheduler.parent.mkdir(parents=True)
    target.write_text("runner\n", encoding="utf-8")
    scheduler.write_text("scheduler\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    state = {
        str(manager.TARGET_FILE): manager.PRE_GENERIC_NUMERICAL_PATCHED_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_PATCHED_SHA256,
    }
    applied = []

    def fake_sha256(path):
        if path == target:
            return state[str(manager.TARGET_FILE)]
        if path == scheduler:
            return state[str(manager.SCHEDULER_FILE)]
        return path.name.ljust(64, "0")[:64]

    def fake_patch(_root, *, patch, reverse, dry_run):
        if not dry_run:
            applied.append((patch, reverse))
            if patch == manager.PRE_GENERIC_NUMERICAL_PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.TIMING_PATCHED_SHA256
            elif patch == manager.TIMING_PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.WORKER_HOOKS_SHA256
            elif patch == manager.SCHEDULER_PATCH and reverse:
                state[str(manager.SCHEDULER_FILE)] = manager.SCHEDULER_BASE_SHA256
            elif patch == manager.PATCH and reverse:
                state[str(manager.TARGET_FILE)] = manager.BASE_SHA256
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(manager, "sha256", fake_sha256)
    monkeypatch.setattr(manager, "run_patch", fake_patch)
    monkeypatch.setattr(manager, "source_commit", lambda _source: manager.BASE_COMMIT)

    assert manager.main(["restore", "--vllm-root", str(root), "--source", str(source)]) == 0
    assert (manager.PRE_GENERIC_NUMERICAL_PATCH, True) in applied
    assert (manager.NUMERICAL_PATCH, True) not in applied
    assert state == {
        str(manager.TARGET_FILE): manager.BASE_SHA256,
        str(manager.SCHEDULER_FILE): manager.SCHEDULER_BASE_SHA256,
    }


def test_gate_helpers_request_explicit_mutually_exclusive_states():
    helper = (
        Path(__file__).resolve().parents[1]
        / "integrations"
        / "vllm"
        / "phase4b1_gate_helpers.sh"
    ).read_text(encoding="utf-8")
    restore = helper.split("phase4b1_restore_stock ()", 1)[1].split(
        "phase4b1_freeze_stock_reference ()", 1
    )[0]
    apply = helper.split("phase4b1_apply_patch_stack ()", 1)[1].split(
        "phase4b1_require_outcome_a ()", 1
    )[0]
    assert "--expect-state stock" in restore
    assert "--expect-state patched" not in restore
    assert "--expect-state patched" in apply
    assert "--expect-state stock" not in apply
    assert "phase4b1_reuse_stock_reference ()" in helper
    assert "reuse_immutable_stock_reference" in helper
