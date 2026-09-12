"""Producer-contract compact evidence: window commits, TP joins and scan exclusions."""

import copy

import pytest

from specrhythm.serving.common import DataError
from specrhythm.serving.decode_scan_plan import options, selected_point
from specrhythm.serving.decode_scan_results import timing
from specrhythm.serving.decode_scan_window import ScanWindow


def evidence(mode, warmup_sequence=None):
    p, opts = selected_point(mode, 16), options()
    grouped = mode == "pingpong"
    w = ScanWindow(opts, 16, grouped)
    final = {
        str(i): dict(
            request_id=str(i),
            generated_token_ids=[10],
            commits=[],
            state="DIAGNOSTIC_CANCELLED",
            cohort=("A" if i < 8 else "B") if grouped else None,
            admission_ns=1 if i < 16 else None,
        )
        for i in range(360)
    }
    devices = [
        dict(
            device=dict(identity=dict(global_rank=r, gpu_uuid=f"GPU-{r + 1}"), forwards=[]),
            rounds=[],
            target_rows=[],
        )
        for r in (0, 1)
    ]
    steps = []
    clock = 10_000
    sequence = warmup_sequence or "ABAB"
    for index in range(len(sequence) + 31 if grouped else 17):
        w.ready(clock, population=dict(
            active_requests=16, request_ids=[str(i) for i in range(16)],
            cohorts={"A": [str(i) for i in range(8)], "B": [str(i) for i in range(8, 16)]}))
        measured = w.start_ns is not None
        cohort = (sequence[index] if index < len(sequence) else
                  "AB"[(index - len(sequence)) % 2]) if grouped else None
        ids = [
            str(i)
            for i in (range(8) if cohort == "A" else range(8, 16) if cohort == "B" else range(16))
        ]
        start = clock
        clock += (990_000_000 if grouped else 2_000_000_000) + 1000
        rows = []
        for rid in ids:
            n = len(final[rid]["generated_token_ids"]) + 2
            k = 0 if mode == "target" else 4
            row = dict(
                request_id=rid,
                internal_request_id=rid,
                context_length=n,
                query_positions=k + 1,
                candidate_positions=k,
                base_root_positions=1,
                position_start=n - 1,
                position_end_exclusive=n + k,
            )
            rows.append(row)
            diag = dict(
                request_id=rid,
                context_length=n,
                query_length=k + 1,
                proposal_token_ids=[11, 12, 13, 14][:k],
                position_ids=list(range(n - 1, n + k)),
                structural_errors=[],
                target_forward_start_ns=start + 1,
            )
            devices[0]["target_rows"].append(diag)
            committed = [11, 99] if k else [99]
            final[rid]["generated_token_ids"].extend(committed)
            final[rid]["commits"].append(dict(token_ids=committed, timestamp_ns=clock))
            if k:
                devices[0]["rounds"].append(
                    dict(
                        request_id=rid,
                        parent_prefix_len=n,
                        proposal_token_ids=diag["proposal_token_ids"],
                        accepted_draft_token_ids=[11],
                        rejected_draft_token_ids=[12, 13, 14],
                        target_correction_token_ids=[99],
                        target_bonus_token_ids=[],
                        committed_token_ids=committed,
                        accepted_draft_tokens=1,
                        rejected_draft_tokens=3,
                    )
                )
        step = dict(
            B=len(ids),
            request_ids=ids,
            cohort=cohort,
            rows=rows,
            start_ns=start,
            end_ns=clock + 1,
            window=measured,
            output_commit_complete=True,
            committed_tokens=len(ids) * (1 if mode == "target" else 2),
            population=dict(held_slots=16, cohort_held={"A": 8, "B": 8}),
        )
        for d in devices:
            d["device"]["forwards"].append(
                dict(
                    host_start_ns=start + 1, internal_request_ids=ids, B=len(ids), gpu_event_ms=0.1
                )
            )
        steps.append(step)
        w.step_completed(step, clock + 1)
        clock += 10_000
    w.window_state["request_ids"] = [str(i) for i in range(16)]
    w.end_ns = clock
    r = dict(
        start_ns=1,
        measurement_start_ns=w.start_ns,
        measurement_end_ns=clock,
        end_ns=clock + 1_000_000,
        stop_reason="time_budget",
        warmup_steps=w.warmup_steps,
        target_steps=steps,
        target_devices=devices,
        requests=list(final.values()),
        prompt_lengths=dict.fromkeys(final, 2),
        decode_scan=w.evidence(),
        events=[],
    )
    return r, dict(fixed_device=dict(identity=dict(gpu_uuid="GPU-0"))), p, opts


@pytest.mark.parametrize("mode", ["target", "serial", "pingpong"])
def test_actual_window_counts_roots_candidates_once_and_uses_actual_elapsed(mode):
    r, b, p, opts = evidence(mode)
    result = timing(r, b, p, opts)
    assert result["measurement_status"] == "PASS"
    steps = result["target_steps"]
    assert steps == (31 if mode == "pingpong" else 15)
    expected = steps * (8 if mode == "pingpong" else 16) * (1 if mode == "target" else 2)
    assert (
        result["committed_window_tokens"] == expected
    )  # no prefill, warmup, rejected, TP duplicate
    assert result["candidate_positions"] == result["accepted_tokens"] + result["rejected_tokens"]
    assert result["decode_throughput_tok_s"] == expected * 1000 / result["measured_window_ms"]
    assert result["window_overrun_ms"] > 0
    assert result["complete_rotations"] == 15
    assert result["partial_rotations"] == (1 if mode == "pingpong" else 0)
    assert result["full_active_time_fraction"] == 1


@pytest.mark.parametrize("damage", ["commit", "root", "TP", "KV", "cohort", "rotation"])
def test_real_accounting_and_identity_failures_still_block(damage):
    r, b, p, opts = evidence("pingpong")
    if damage == "commit":
        r["requests"][0]["commits"][-1]["token_ids"] *= 2
    elif damage == "root":
        r["target_steps"][-1]["rows"][0]["base_root_positions"] = 2
    elif damage == "TP":
        r["target_devices"][1]["device"]["forwards"].pop()
    elif damage == "KV":
        r["target_devices"][0]["target_rows"][-1]["structural_errors"] = ["private KV changed"]
    elif damage == "cohort":
        r["requests"][0]["cohort"] = "B"
    else:
        r["decode_scan"]["complete_rotations"].pop()
    with pytest.raises(DataError):
        timing(r, b, p, opts)


def test_clean_pool_exhaustion_and_partial_prevention_keep_evidence_but_exclude():
    r, b, p, opts = evidence("serial")
    r["stop_reason"] = "pool_exhausted_before_window"
    result = timing(r, b, p, opts)
    assert result["measurement_status"] == "INSUFFICIENT" and result["measurement_available"]
    assert not result["formal_comparison_eligible"]
    r["stop_reason"] = "partial_batch_prevented"
    r["decode_scan"]["rejected_step"] = dict(
        expected_batch=16, scheduled_batch=15, model_forward_issued=False
    )
    assert timing(r, b, p, opts)["rejected_step"]["scheduled_batch"] == 15
    damaged = copy.deepcopy(r)
    damaged["target_steps"][-1]["B"] = 15
    with pytest.raises(DataError, match="batch"):
        timing(damaged, b, p, opts)


@pytest.mark.parametrize("sequence", ["ABAB", "ABAAB", "BABBA"])
def test_real_window_to_qualifier_accepts_closed_warmup_after_historical_extra(sequence):
    # Actual producer state, TP forwards/commits, and production timing validator.
    # ABAAB reproduces f718: ready() opens at two rotations/pending=None,
    # rotations() returns (2, 1), and timing() falsely rejects the closed boundary.
    r, b, p, opts = evidence("pingpong", sequence)
    assert r["warmup_steps"] == len(sequence)
    assert r["decode_scan"]["warmup_rotations"] == 2
    result = timing(r, b, p, opts)
    assert result["measurement_status"] == "PASS"
    assert result["target_steps"] == 31
    assert result["committed_window_tokens"] == 31 * 8 * 2
    assert result["partial_rotations"] == 1  # Legal end half, actual time/tokens.
    boundary = result["scan_warmup_boundary"]
    assert boundary["pending_step"] is None
    assert boundary["historical_unpaired_steps"] == ([] if sequence == "ABAB" else [3])
    assert "".join(s["cohort"] for s in boundary["steps"]) == sequence
    assert boundary["committed_tokens_excluded"] == len(sequence) * 8 * 2
    assert all(s["end_ns"] <= r["measurement_start_ns"] for s in boundary["steps"])
    assert all("request_ids" not in s and "token_ids" not in s for s in boundary["steps"])


@pytest.mark.parametrize("damage, message", [
    ("missing", "mandatory warmup boundary"),
    ("extra", "boundary evidence differs"),
    ("time", "Target TP actual request-row"),
    ("tokens", "warmup committed token/time"),
    ("active", "full warmup/population"),
    ("cohort_size", "full warmup/population"),
    ("lifecycle", "active identities"),
    ("half", "pending warmup half-rotation"),
])
def test_warmup_boundary_material_evidence_is_checked(damage, message):
    r, b, p, opts = evidence("pingpong", "ABAAB")
    boundary = r["decode_scan"]["warmup_boundary"]
    if damage == "missing":
        del r["decode_scan"]["warmup_boundary"]
    elif damage == "extra":
        boundary["historical_unpaired_steps"] = []
    elif damage == "time":
        r["target_steps"][2]["start_ns"] = r["target_steps"][1]["start_ns"]
    elif damage == "tokens":
        r["target_steps"][2]["committed_tokens"] += 1
    elif damage == "active":
        r["decode_scan"]["window_initial_population"]["active_requests"] -= 1
    elif damage == "cohort_size":
        pop = r["decode_scan"]["window_initial_population"]
        pop["cohorts"]["B"].append(pop["cohorts"]["A"].pop())
    elif damage == "lifecycle":
        r["requests"][0]["completion_ns"] = r["measurement_start_ns"] - 1
    else:
        first = next(s for s in r["target_steps"] if s["window"])
        first["window"] = False
        r["warmup_steps"] += 1
        r["measurement_start_ns"] = first["end_ns"] + 1
        r["decode_scan"]["partial_rotations"] = []
    with pytest.raises(DataError, match=message):
        timing(r, b, p, opts)
