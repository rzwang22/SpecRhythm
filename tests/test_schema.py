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
        ("input_tokens", 1.5),
        ("input_tokens", True),
        ("output_tokens", 0),
        ("output_tokens", 2.5),
        ("output_tokens", False),
        ("slo_tpot_ms", 0),
        ("slo_tpot_ms", True),
        ("slo_tpot_ms", float("inf")),
        ("acceptance_probability", 1.1),
        ("acceptance_probability", True),
        ("acceptance_probability", float("nan")),
        ("draft_confidence", 1.1),
        ("draft_confidence", True),
        ("draft_confidence", float("nan")),
        ("arrival_time_ms", True),
        ("arrival_time_ms", float("inf")),
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


@pytest.mark.parametrize("request_id", ["", "   ", None, 3])
def test_request_id_must_be_a_non_empty_string(request_id):
    with pytest.raises(ValueError):
        WorkloadRequest(request_id, 0, 1, 1, 40)


@pytest.mark.parametrize("turn_index", [-1, 1.5, True])
def test_turn_index_must_be_a_non_negative_integer(turn_index):
    with pytest.raises(ValueError):
        WorkloadRequest("r1", 0, 1, 1, 40, turn_index=turn_index)
