from specrhythm.diagnostics import (
    request_level_shaping_loss_report,
    shaping_loss_report,
)


def test_shaping_loss_decomposes_numerator_denominator_and_class_transfer():
    baseline = {
        "slo_good_tokens": 100,
        "makespan_ms": 1000,
        "goodput_tokens_per_s": 100,
        "slo_class_metrics": {
            "40": {
                "drafted_tokens": 10,
                "accepted_tokens": 5,
                "mean_expected_progress": 1,
                "scheduled_request_ratio": 0.5,
                "attainment": 0.5,
                "requests": 2,
            }
        },
    }
    shaped = {
        "slo_good_tokens": 120,
        "makespan_ms": 1100,
        "goodput_tokens_per_s": 109,
        "slo_class_metrics": {
            "40": {
                "drafted_tokens": 14,
                "accepted_tokens": 8,
                "mean_expected_progress": 2,
                "scheduled_request_ratio": 1,
                "attainment": 1,
                "requests": 2,
            }
        },
    }
    report = shaping_loss_report(baseline, shaped)
    assert report["slo_good_tokens_delta"] == 20
    assert report["makespan_ms_delta"] == 100
    assert report["class_deltas"]["40"]["candidate_budget_delta"] == 4
    assert report["class_deltas"]["40"]["realized_progress_delta"] == 3


def test_request_level_shaping_loss_counts_harmed_and_unrescued_requests():
    baseline = {
        "request_allocation_diagnostics": [
            {
                "request_id": "tight",
                "slo_tpot_ms": 40,
                "allocated_candidate_nodes": 1,
                "expected_progress": 0.5,
                "realized_candidate_progress": 0,
                "attained": False,
            },
            {
                "request_id": "relaxed",
                "slo_tpot_ms": 150,
                "allocated_candidate_nodes": 4,
                "expected_progress": 2,
                "realized_candidate_progress": 2,
                "attained": True,
            },
        ]
    }
    shaped = {
        "request_allocation_diagnostics": [
            {
                "request_id": "tight",
                "slo_tpot_ms": 40,
                "allocated_candidate_nodes": 3,
                "expected_progress": 1.5,
                "realized_candidate_progress": 1,
                "attained": False,
            },
            {
                "request_id": "relaxed",
                "slo_tpot_ms": 150,
                "allocated_candidate_nodes": 2,
                "expected_progress": 1,
                "realized_candidate_progress": 1,
                "attained": False,
            },
        ]
    }
    report = request_level_shaping_loss_report(baseline, shaped)
    assert report["tight_extra_budget_but_still_missed"] == 1
    assert report["relaxed_150ms_lost_attainment"] == 1
