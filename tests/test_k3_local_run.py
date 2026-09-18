"""Real inner Bash -> local archive -> verified durable copy; GPU commands alone replaced."""

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest
from test_k3_report_publication import payload

from specrhythm.serving import k3_local_run as local
from specrhythm.serving.k3 import MODES
from specrhythm.serving.ping_prepost_delivery import export

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("configuration,batch", [("k3-b16-v1", 16), ("k3-b64-v1", 64)])
@pytest.mark.parametrize("copy_fault", ["none", "write", "digest", "archive"])
def test_actual_runner_first_error_one_archive_and_local_fallback(
    tmp_path, monkeypatch, capsys, copy_fault, configuration, batch
):
    repo, binary = tmp_path / "repo", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    binary.mkdir()
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    shutil.copyfile(REPO / f"scripts/run_k3_b{batch}.sh", repo / f"scripts/run_k3_b{batch}.sh")
    sha = "a" * 40
    (binary / "git").write_text(
        '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo ' + sha + "; fi\n"
    )
    (binary / "git").chmod(0o755)
    (repo / "scripts/run_decode_scan.sh").write_text("""#!/bin/bash
mkdir -p "$SR_FIXED_ROOT/runs/injected"
printf '{"unfinished":' > "$SR_FIXED_ROOT/runs/injected/draft-backend-report.json"
printf 'original injected prepare error\\n' > "$SR_FIXED_ROOT/runs/injected/target.log"
printf 'FIRST CPU INJECTED ERROR mode=%s\\n' "$SR_AUDIT_SERVING_MODE"
exit 23
""")
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SR_FIXED_PYTHON", sys.executable)
    monkeypatch.delenv("SR_AUDIT_SERVING_MODE", raising=False)
    original_copy, original_digest = local.shutil.copyfileobj, local.sha256_file

    def copying(source, target, length):
        if copy_fault == "write":
            target.write(source.read(32))
            raise OSError("injected DPC copy failure; no claim about real DPC")
        return original_copy(source, target, length)

    def digest(path):
        if copy_fault == "digest" and str(path).endswith(".copying"):
            return "0" * 64
        return original_digest(path)

    if copy_fault == "archive":
        def interrupted_add(*args, **kwargs):
            raise OSError("injected archive write failure")
        monkeypatch.setattr(tarfile.TarFile, "addfile", interrupted_add)
    monkeypatch.setattr(local.shutil, "copyfileobj", copying)
    monkeypatch.setattr(local, "sha256_file", digest)
    code = local.run(
        repo,
        sha,
        local_base=tmp_path / "local",
        persistent=tmp_path / "durable",
        tag="case",
        minimum_free_bytes=1, configuration=configuration,
    )
    output = capsys.readouterr().out
    assert code == 23  # Not 41 (bad JSON), 43 (copy), or a synthetic success.
    if copy_fault == "archive":
        assert "UPLOAD ONLY:" not in output and "NO ARCHIVE: retained evidence directory" in output
        state = json.loads((tmp_path / "local/case/delivery-status.json").read_text())
        assert state["final_exit_code"] == 23 and state["export_exit_code"] == 41
        assert state["archive_error"] and state["upload_path"] is None
        return
    assert output.count("UPLOAD ONLY:") == 1
    assert "FIRST CPU INJECTED ERROR mode=serial-k3" in output
    upload = Path(output.split("UPLOAD ONLY: ")[1].strip())
    assert local.validate_archive(upload)["archive_integrity"] == "VERIFIED"
    raw, inventory = payload(upload, "points/serial-k3/runs/injected/draft-backend-report.json")
    assert raw == b'{"unfinished":' and inventory["first_exit_code"] == 23
    if batch == 64:
        policy = json.loads(payload(upload, "validation-plan.json")[0])
        assert policy["validation_profile"] == inventory["validation_profile"] == (
            "performance-exploration")
        assert policy["full_output_comparison_planned"] is False
        assert inventory["intentionally_not_run"][0]["path"] == "joint/"
    else:
        assert inventory["validation_profile"] == "strict-output"
    assert (
        inventory["failed_stage"] == "prepare" and inventory["export_validation_exit_code"] == 41
    )
    root = tmp_path / "local/case"
    assert root.exists() and len(list(root.glob("*.tar.gz"))) == 1
    state = json.loads((root / "delivery-status.json").read_text())
    assert state["inner_exit_code"] == state["final_exit_code"] == 23
    assert state["export_exit_code"] == 41
    assert not any((root / "pingpong-k3-delivery-case/points" / m).exists() for m in MODES[1:])
    if copy_fault == "none":
        assert upload.parent == tmp_path / "durable" and state["delivery_exit_code"] == 0
        assert state["persistent_copy"]["sha256"] == local.sha256_file(root / upload.name)
    else:
        assert upload.parent == root and state["delivery_exit_code"] == 43
        packaged_state = json.loads(payload(upload, "delivery-status.json")[0])
        assert packaged_state["status"] == "LOCAL_FALLBACK"
        assert packaged_state["delivery_error"]
        assert not list((tmp_path / "durable").glob("*.tar.gz"))


def test_paths_resolved_storage_facts_retention_and_collision(tmp_path, monkeypatch):
    base = tmp_path / "real"
    base.mkdir()
    link = tmp_path / "link"
    link.symlink_to(base, target_is_directory=True)
    root, durable, facts = local.prepare_local(
        link, tmp_path / "durable", "tag", minimum_free_bytes=1
    )
    assert root == base / "tag" and facts["resolved_local_base"] == str(base)
    assert facts["local_filesystem"]["type"] and "not a general" in facts["writable_probe"]
    assert "NVMe" in facts["physical_medium"] and facts["retention"].startswith("KEEP")
    with pytest.raises(FileExistsError):
        local.prepare_local(link, durable, "tag", minimum_free_bytes=1)
    with pytest.raises(ValueError, match="free bytes"):
        local.prepare_local(link, durable, "space", minimum_free_bytes=10**30)
    monkeypatch.setattr(
        local, "filesystem", lambda p: dict(type="dpc", source="injected", mount="/")
    )
    with pytest.raises(ValueError, match="resolves to remote"):
        local.prepare_local(link, durable, "remote", minimum_free_bytes=1)
    assert not (base / "remote").exists()


def test_unique_copy_never_overwrites_history(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    archive = tmp_path / "archive.tar.gz"
    export(directory, archive, first_code=23, modes=MODES)
    expected = local.validate_archive(archive)
    durable = tmp_path / "durable"
    durable.mkdir()
    first = local.deliver(archive, durable, expected)
    with pytest.raises(FileExistsError):
        local.deliver(archive, durable, expected)
    assert local.sha256_file(Path(first["path"])) == expected["sha256"]


@pytest.mark.parametrize('batch', [16, 64])
def test_public_entry_routes_before_any_mutable_run_directory(batch):
    text = (REPO / f"scripts/run_k3_b{batch}.sh").read_text()
    assert text.index("specrhythm.serving.k3_local_run") < text.index(
        'mkdir -p "$SR_PING_DELIVERY'
    )
    assert 'export SR_PING_DELIVERY="$SR_K3_LOCAL_DELIVERY"' in text
    assert "SR_K3_MANAGED_LOCAL" in text and "PY_OUTCOME" in text
