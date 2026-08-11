import json
from collections import Counter
from pathlib import Path

from specrhythm.cli import main
from specrhythm.provenance import sha256_file
from specrhythm.validation import validate_workload
from specrhythm.workload import (
    generate_replay_workload,
    load_json,
    select_arrival_replay,
)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mooncake-arrivals.jsonl"
CONFIG = ROOT / "configs" / "workloads" / "r3-mooncake-622-proxy.json"


def _generate(tmp_path: Path, name: str = "workload.jsonl") -> Path:
    output = tmp_path / name
    result = main(
        [
            "generate",
            "--config",
            str(CONFIG),
            "--arrival-trace",
            str(FIXTURE),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    return output


def test_replay_is_byte_deterministic_and_adds_no_requests(tmp_path):
    first = _generate(tmp_path, "first.jsonl")
    second = _generate(tmp_path, "second.jsonl")
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text().splitlines()) == len(FIXTURE.read_text().splitlines())

    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert all(record["conversation_id"] is None for record in records)
    assert all(record["turn_index"] is None for record in records)


def test_chronological_window_and_time_scaling():
    replay = select_arrival_replay(
        FIXTURE,
        window_start_ms=500,
        window_duration_ms=2000,
        time_scale=2,
    )
    assert replay.source_timestamps_ms == (
        550,
        700,
        900,
        1000,
        1100,
        1250,
        1500,
        1800,
        2000,
        2300,
    )
    assert replay.arrival_times_ms == (0, 75, 175, 225, 275, 350, 475, 625, 725, 875)


def test_replay_orders_source_timestamps_chronologically(tmp_path):
    source = tmp_path / "unordered.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"timestamp": timestamp})
            for timestamp in (900, 550, 700)
        )
        + "\n"
    )
    replay = select_arrival_replay(source, window_start_ms=500, window_duration_ms=500)
    assert replay.source_timestamps_ms == (550, 700, 900)
    assert replay.arrival_times_ms == (0, 150, 350)


def test_r3_task_and_slo_mixture_is_622():
    replay = select_arrival_replay(FIXTURE)
    workload = generate_replay_workload(load_json(CONFIG), replay)
    task_counts = Counter(request.task for request in workload.requests)
    slo_counts = Counter(request.slo_tpot_ms for request in workload.requests)
    assert task_counts == {"code": 18, "chat": 6, "summarization": 6}
    assert slo_counts == {40.0: 18, 50.0: 6, 150.0: 6}


def test_validator_accepts_valid_replay(tmp_path):
    output = _generate(tmp_path)
    report = validate_workload(
        output,
        config=load_json(CONFIG),
        arrival_trace_path=FIXTURE,
    )
    assert report["valid"] is True
    assert report["summary"]["request_count"] == 30
    assert report["summary"]["task_proportions"] == {
        "chat": 0.2,
        "code": 0.6,
        "summarization": 0.2,
    }


def test_manifest_checksums_are_correct(tmp_path):
    output = tmp_path / "workload.jsonl"
    manifest_path = tmp_path / "manifest.json"
    result = main(
        [
            "generate",
            "--config",
            str(CONFIG),
            "--arrival-trace",
            str(FIXTURE),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--source-commit-sha",
            "0123456789abcdef",
        ]
    )
    assert result == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_trace_sha256"] == sha256_file(FIXTURE)
    assert manifest["config_sha256"] == sha256_file(CONFIG)
    assert manifest["output_workload_sha256"] == sha256_file(output)
    assert manifest["request_count"] == 30
    assert not Path(manifest["source_trace_path"]).is_absolute()
