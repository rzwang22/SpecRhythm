import json

import pytest

from specrhythm.schema import Workload, WorkloadRequest


def test_workload_round_trip(tmp_path):
    request = WorkloadRequest(
        request_id="r1",
        arrival_time_ms=10.0,
        input_tokens=32,
        output_tokens=8,
        slo_tpot_ms=40.0,
        task="code",
        metadata={"nested": [1, 2]},
    )
    path = tmp_path / "workload.jsonl"
    Workload([request]).save_jsonl(path)

    loaded = Workload.load_jsonl(path)
    assert loaded.requests == [request]
    assert json.loads(path.read_text().strip())["request_id"] == "r1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("arrival_time_ms", -1),
        ("input_tokens", 0),
        ("output_tokens", 0),
        ("slo_tpot_ms", 0),
        ("acceptance_probability", 1.1),
    ],
)
def test_invalid_request_is_rejected(field, value):
    values = {
        "request_id": "r1",
        "arrival_time_ms": 0,
        "input_tokens": 1,
        "output_tokens": 1,
        "slo_tpot_ms": 40,
        "acceptance_probability": 0.5,
    }
    values[field] = value
    with pytest.raises(ValueError):
        WorkloadRequest(**values)
