import json

import pytest

from specrhythm.cli import main
from specrhythm.validation import validate_workload


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
