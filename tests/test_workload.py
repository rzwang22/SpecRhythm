import json

from specrhythm.workload import generate_workload, import_mooncake, summarize_workload


def _config():
    return {
        "seed": 3,
        "arrival_segments": [{"duration_s": 5, "rate_per_s": 3, "cv": 1.5}],
        "client_count": 4,
        "slo_mode": "task",
        "task_profiles": [
            {
                "name": "test",
                "weight": 1,
                "slo_tpot_ms": 40,
                "input_median": 16,
                "input_sigma": 0.1,
                "output_median": 8,
                "output_sigma": 0.1,
                "length_correlation": 0.5,
                "acceptance_probability": 0.75,
            }
        ],
        "conversation": {"start_probability": 0},
    }


def test_generation_is_deterministic_and_sorted():
    first = generate_workload(_config())
    second = generate_workload(_config())
    assert first.requests == second.requests
    arrivals = [request.arrival_time_ms for request in first.requests]
    assert arrivals == sorted(arrivals)
    assert summarize_workload(first)["requests"] > 0


def test_trace_composition_uses_supplied_timestamps():
    workload = generate_workload(_config(), [100, 20, 50])
    assert [request.arrival_time_ms for request in workload.requests] == [20, 50, 100]


def test_summary_rate_uses_observed_inter_arrivals():
    workload = generate_workload(_config(), [100, 600, 1100])
    summary = summarize_workload(workload)
    assert summary["duration_s"] == 1.0
    assert summary["mean_rate_per_s"] == 2.0


def test_mooncake_import_preserves_lengths_and_hashes(tmp_path):
    source = tmp_path / "mooncake.jsonl"
    records = [
        {"timestamp": 100, "input_length": 512, "output_length": 7, "hash_ids": [1]},
        {"timestamp": 300, "input_length": 1024, "output_length": 9, "hash_ids": [1, 2]},
    ]
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    workload = import_mooncake(source, time_scale=2)
    assert [request.arrival_time_ms for request in workload.requests] == [0, 100]
    assert workload.requests[1].input_tokens == 1024
    assert workload.requests[1].metadata["prefix_block_hash_ids"] == [1, 2]
