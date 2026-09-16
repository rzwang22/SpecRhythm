"""Real child Bash, policy, failure-summary, flat exporter and local delivery."""

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

from specrhythm.phase4.manifest import atomic_write_json
from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_local_run import run, validate_archive
from specrhythm.serving.k3_repeat_run import CONFIGS, REPEATS, declaration

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("configuration", CONFIGS)
def test_real_repeated_entry_first_error_one_archive(configuration, tmp_path, monkeypatch, capsys):
    repo, binary = tmp_path / "repo", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    binary.mkdir()
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    shutil.copyfile(REPO / "scripts/run_k3_b64.sh", repo / "scripts/run_k3_b64.sh")
    sha = "a" * 40
    (binary / "git").write_text(
        '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo ' + sha + "; fi\n")
    (binary / "git").chmod(0o755)
    (repo / "scripts/run_decode_scan.sh").write_text('''#!/bin/bash
printf '%s %s %s\\n' "$1" "$SR_K3_OBSERVATION" "$SR_K3_DRAFT_DISPATCH" >> "$CALLS"
mkdir -p "$SR_FIXED_ROOT/runs/injected"
printf '{"unfinished":' > "$SR_FIXED_ROOT/runs/injected/draft-backend-report.json"
printf 'first injected GPU prepare failure\\n' > "$SR_FIXED_ROOT/runs/injected/target.log"
exit 23
''')
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SR_FIXED_PYTHON", sys.executable)
    monkeypatch.setenv("SR_K3_EXPERIMENT", configuration)
    monkeypatch.setenv("CALLS", str(tmp_path / "calls"))
    for key in ("SR_K3_REPEAT_CHILD", "SR_K3_LOCAL_DELIVERY", "SR_K3_DRAFT_DISPATCH"):
        monkeypatch.delenv(key, raising=False)
    rc = run(repo, sha, local_base=tmp_path / "local", persistent=tmp_path / "durable",
             tag="case", minimum_free_bytes=1, configuration="k3-b64-v1")
    assert rc == 23
    output = capsys.readouterr().out
    assert output.count("UPLOAD ONLY:") == 1
    upload = Path(output.split("UPLOAD ONLY: ")[1].strip())
    assert validate_archive(upload)["archive_integrity"] == "VERIFIED"
    assert (tmp_path / "calls").read_text().splitlines() == [
        "prepare deferred-window " + CONFIGS[configuration]]
    with tarfile.open(upload) as t:
        inventory = json.load(t.extractfile("inventory.json"))
        assert inventory["first_exit_code"] == 23 and inventory["failed_stage"] == "prepare"
        assert inventory["output_equivalence_status"] == "NOT_RUN"
        paths = inventory["logical_paths"]
        first = json.load(t.extractfile(paths["first-failure.json"]))
        assert "serial-k3" in json.dumps(first)
        assert "0-forward" in json.dumps(first)
        raw = "repeats/0-forward/points/serial-k3/runs/injected/draft-backend-report.json"
        assert t.extractfile(paths[raw]).read() == b'{"unfinished":'
        assert not any(p.startswith("repeats/1-reverse/points/") for p in paths)
        assert any(c.get("status") == "NOT_STARTED" for c in inventory["children"])


def test_repeat_controller_orders_and_stops_without_retry(tmp_path, monkeypatch):
    from specrhythm.serving import k3_repeat_run as repeat

    seen = []

    def child(argv, env):
        seen.append((env["SR_K3_REVERSE"], env["SR_K3_DRAFT_DISPATCH"],
                     env["SR_K3_LOCAL_DELIVERY"]))
        root = Path(env["SR_K3_LOCAL_DELIVERY"])
        root.mkdir()
        atomic_write_json(root / "runner-outcome.json", dict(
            first_exit_code=0 if len(seen) == 1 else 23, stage="execution", mode=MODES[-1]))
        return 0 if len(seen) == 1 else 23

    # Only the external per-repeat GPU runner is replaced. Real controller,
    # manifests, failure code and compare-on-incomplete-results all run.
    monkeypatch.setattr(repeat.subprocess, "call", child)
    root = tmp_path / "delivery"
    assert repeat.run(root, tmp_path, "a" * 40, "unified") == 23
    assert [x[:2] for x in seen] == [("0", "unified"), ("1", "unified")]
    assert len({x[2] for x in seen}) == 2
    spec = json.loads((root / "experiment-plan.json").read_text())
    assert spec == declaration("unified")
    assert [r["name"] for r in spec["repeats"]] == list(REPEATS)
    assert spec["repeats"][0]["modes"] == list(MODES)
    assert spec["repeats"][1]["modes"] == list(reversed(MODES))
    assert json.loads((root / "comparison.json").read_text())["valid"] is False


@pytest.mark.parametrize("configuration", CONFIGS)
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_bash_runs_declared_order_with_same_io(
    tmp_path, monkeypatch, configuration, reverse
):
    import test_k3_runbook as prior

    monkeypatch.setenv("SR_K3_REVERSE", "1" if reverse else "0")
    monkeypatch.setenv("SR_K3_OBSERVATION", "deferred-window")
    monkeypatch.setenv("SR_K3_DRAFT_DISPATCH", CONFIGS[configuration])
    monkeypatch.delenv("SR_K3_EXPERIMENT", raising=False)
    if reverse:
        monkeypatch.setattr(prior, "MODES", tuple(reversed(MODES)))
    prior.test_one_bundle_first_error_stops_points_and_parent_remains_open(
        tmp_path, "none", "", False, "run_k3_b64.sh")


def test_broken_declaration_retains_raw_bytes_and_original_code(tmp_path):
    from specrhythm.serving.ping_prepost_delivery import export

    root = tmp_path / "broken"
    root.mkdir()
    (root / "experiment-plan.json").write_bytes(b'{"configuration":')
    output = tmp_path / "broken.tar.gz"
    result = export(root, output, first_code=23, stage="prepare", modes=MODES)
    assert result["export_validation_exit_code"] == 41 and result["first_exit_code"] == 23
    with tarfile.open(output) as t:
        assert t.extractfile(result["logical_paths"]["experiment-plan.json"]).read() == (
            b'{"configuration":')
    assert validate_archive(output)["archive_integrity"] == "VERIFIED"
