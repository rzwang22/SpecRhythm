import json
from pathlib import Path

import pytest

from specrhythm.cli import main
from specrhythm.validation import validate_workload
from specrhythm.workload import load_json

ROOT = Path(__file__).parents[1]
R3_CONFIG = load_json(ROOT / "configs" / "workloads" / "r3-mooncake-622-proxy.json")


def _record(request_id, arrival_time_ms, acceptance_probability=0.7):
    return {
        "request_id": request_id,
        "arrival_time_ms": arrival_time_ms,
        "input_tokens": 10,
        "output_tokens": 5,
        "slo_tpot_ms": 40,
        "task": "code",
        "acceptance_probability": acceptance_probability,
    }


def _write(tmp_path, records):
    path = tmp_path / "workload.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


@pytest.mark.parametrize(
    ("records", "error_code"),
    [
        (
            [_record("request-0", 10), _record("request-1", 5)],
            "non_monotonic_arrival",
        ),
        (
            [_record("request-0", 0), _record("request-0", 5)],
            "duplicate_request_id",
        ),
        (
            [_record("request-0", 0, acceptance_probability=1.1)],
            "invalid_acceptance_probability",
        ),
    ],
)
def test_validator_rejects_invalid_workload(tmp_path, records, error_code):
    path = _write(tmp_path, records)
    report = validate_workload(path)
    assert report["valid"] is False
    assert error_code in {error["code"] for error in report["errors"]}


def test_validate_cli_returns_nonzero_and_writes_report(tmp_path):
    path = _write(tmp_path, [_record("request-0", 0, acceptance_probability=-0.1)])
    report_path = tmp_path / "validation.json"
    result = main(
        ["validate", "--workload", str(path), "--output", str(report_path)]
    )
    assert result == 1
    assert json.loads(report_path.read_text())["valid"] is False


def test_validator_rejects_empty_workload(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    report = validate_workload(path)
    assert report["valid"] is False
    assert {error["code"] for error in report["errors"]} == {"empty_workload"}


def test_validator_rejects_wrong_r3_task_and_slo_mixture(tmp_path):
    records = [_record(f"request-{index}", index * 100) for index in range(30)]
    report = validate_workload(_write(tmp_path, records), config=R3_CONFIG)
    error_codes = {error["code"] for error in report["errors"]}
    assert report["valid"] is False
    assert "r3_task_mixture_mismatch" in error_codes
    assert "r3_slo_mixture_mismatch" in error_codes


def test_validator_rejects_r3_task_slo_mismatch(tmp_path):
    tasks = ["code"] * 18 + ["chat"] * 6 + ["summarization"] * 6
    slos = {"code": 40, "chat": 50, "summarization": 150}
    records = []
    for index, task in enumerate(tasks):
        record = _record(f"request-{index}", index * 100)
        record["task"] = task
        record["slo_tpot_ms"] = slos[task]
        records.append(record)
    records[-1]["slo_tpot_ms"] = 40

    report = validate_workload(_write(tmp_path, records), config=R3_CONFIG)
    error_codes = {error["code"] for error in report["errors"]}
    assert report["valid"] is False
    assert "r3_task_slo_mismatch" in error_codes
    assert "r3_slo_mixture_mismatch" in error_codes


def test_validator_reports_observed_and_window_rates(tmp_path):
    records = [_record("request-0", 0), _record("request-1", 500), _record("request-2", 1000)]
    report = validate_workload(
        _write(tmp_path, records),
        window_duration_ms=2000,
        time_scale=2,
    )
    assert report["valid"] is True
    assert report["summary"]["observed_iat_rate_per_s"] == 2.0
    assert report["summary"]["window_offered_rate_per_s"] == 3.0
