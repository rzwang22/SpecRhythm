"""Regress B16 settlement batching and equality with the ordinary Serial backend.

Real baseline/eager machines and backends use independently stored, fenced CPU KV.
Call counts describe this controlled case; they are not timings or GPU measurements.
"""

from collections import Counter
from types import SimpleNamespace

from test_rolling_eager_gpu_backend import FencedWorker, greedy_tokens
from test_serial_eager_owner import proposal_row, verify_row

from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.eager_machine import EagerSerialMachine


def test_b16_mixed_settlement_repairs_one_batch_and_reuses_promotions():
    prefixes = {f"request-{index:02d}": (10, 20 + index) for index in range(16)}
    request_ids = tuple(prefixes)
    promoted_ids, rejected_ids, mismatch_ids = request_ids[:5], request_ids[5:13], request_ids[13:]
    repair_ids = rejected_ids + mismatch_ids
    config = SimpleNamespace(max_model_len=4096)
    serial_worker, eager_worker = FencedWorker(), FencedWorker()
    serial_backend = VllmBatchedDraftBackend(config, worker=serial_worker)
    eager_backend = RollingVllmDraftBackend(config, worker=eager_worker)
    serial = BatchedDraftStateMachine(serial_backend, candidate_budget=4)
    eager = EagerSerialMachine(eager_backend, request_ids=request_ids)
    try:
        proposals = []
        for machine in (serial, eager):
            for rid, prefix in prefixes.items():
                machine.initialize(rid, prefix, token_prefix_hash(prefix))
            result = machine.batch_propose([
                proposal_row(rid, prefix) for rid, prefix in prefixes.items()
            ])
            proposals.append({row["request_id"]: row for row in result["proposals"]})
        serial_proposals, eager_proposals = proposals
        for rid in request_ids:
            assert serial_proposals[rid]["proposal_token_ids"] == (
                eager_proposals[rid]["proposal_token_ids"]
            )
        for worker in (serial_worker, eager_worker):
            assert [len(rows) for purpose, rows in worker.calls if purpose == "proposal"] == (
                [16] * 3
            )

        eager.verify_start([verify_row(eager_proposals[rid]) for rid in request_ids])
        # Complete one bridge plus four candidates for all requests before feedback.
        for _ in range(5):
            assert eager.step()
        assert not eager.step()
        eager_rows = [rows for purpose, rows in eager_worker.calls if purpose == "eager"]
        assert [len(rows) for rows in eager_rows] == [16] * 5
        assert all(len(row.suffix) == 1 for rows in eager_rows for row in rows)

        feedback, final_prefixes = [], {}
        for rid in request_ids:
            tokens = tuple(eager_proposals[rid]["proposal_token_ids"])
            if rid in rejected_ids:
                delta = tokens[:2] + (900,)
            elif rid in mismatch_ids:
                delta = tokens + (901,)
            else:
                delta = tokens + greedy_tokens(prefixes[rid] + tokens, 1)
            final_prefixes[rid] = prefixes[rid] + delta
            feedback.append({
                "request_id": rid, "round_id": 0, "committed_delta": list(delta),
                "committed_prefix_hash": token_prefix_hash(final_prefixes[rid]),
                "terminal": False,
            })
        next_rows = [proposal_row(
            rid, final_prefixes[rid], round_id=1,
            remaining=100 - len(final_prefixes[rid]) + len(prefixes[rid]),
        ) for rid in request_ids]

        serial_start, eager_start = len(serial_worker.calls), len(eager_worker.calls)
        before_syncs = Counter(eager_backend.metrics.syncs)
        serial_result = serial.synchronize_and_batch_propose(feedback, next_rows)
        eager.prepare_synchronizations(feedback)
        eager_sync = eager.finish_synchronizations(feedback)
        assert eager_sync is not None and len(eager_sync) == len(serial_result["synchronizations"])
        eager_result = eager.batch_propose(next_rows)

        serial_calls = serial_worker.calls[serial_start:]
        eager_calls = eager_worker.calls[eager_start:]
        assert [(purpose, len(rows)) for purpose, rows in serial_calls] == (
            [("commit", 16)] + [("proposal", 16)] * 3
        )
        # One repair batch excludes the five already materialized promotions.
        assert [(purpose, len(rows)) for purpose, rows in eager_calls] == (
            [("commit", 11)] + [("proposal", 11)] * 3
        )
        repairs = [row for purpose, rows in eager_calls if purpose == "commit" for row in rows]
        assert [row.request_id for row in repairs] == [f"sr-draft:{rid}" for rid in repair_ids]
        assert all(row.context == final_prefixes[rid] and len(row.suffix) == 1
                   for rid, row in zip(repair_ids, repairs))
        assert eager_backend.metrics.syncs - before_syncs == Counter({
            "eager_abort": 11, "eager_parent_settlement": 1, "eager_rebase_complete": 1,
            "bulk_token_d2h": 4, "proposal_complete": 1,
        })
        assert serial_backend.metrics.syncs["commit_complete"] == 1
        assert eager_backend.metrics.syncs["eager_step_complete"] == 5
        assert serial_backend.report()["draft_model_forward_count"] == 7
        assert eager_backend.report()["draft_model_forward_count"] == 12

        serial_next = {row["request_id"]: row for row in serial_result["proposals"]}
        eager_next = {row["request_id"]: row for row in eager_result["proposals"]}
        assert set(serial_next) == set(eager_next) == set(request_ids)
        for rid, final in final_prefixes.items():
            expected = greedy_tokens(final, 4)
            assert tuple(serial_next[rid]["proposal_token_ids"]) == expected
            assert tuple(eager_next[rid]["proposal_token_ids"]) == expected
            source = eager_next[rid]["runtime_provenance"].get("source_continuation_id")
            assert bool(source) == (rid in promoted_ids)
            for machine, worker in ((serial, serial_worker), (eager, eager_worker)):
                physical = machine.backend.states[rid]
                assert machine.requests[rid].committed_token_ids == physical.prefix == final
                assert physical.next_round == 1 and physical.proposal == expected
                assert physical.materialized == len(final) + 3
                assert tuple(worker.memory[physical.internal_id][:physical.materialized]) == (
                    final + expected[:-1]
                )
                assert not worker.inflight
            assert eager.core.state(rid).committed_prefix == final

        counters = eager.counters
        assert counters["committed_tokens"] == sum(len(row["committed_delta"]) for row in feedback)
        assert counters["committed_tokens"] == 64 == (
            counters["parent_accepted_tokens"] + counters["correction_tokens"]
            + counters["bonus_tokens"]
        )
        assert counters["parent_accepted_tokens"] == 48
        assert counters["correction_tokens"] == counters["bonus_tokens"] == 8
        assert counters["promotions"] == 5 and counters["bridge_mismatches"] == 3
        assert counters["parent_rejections"] == 8 and counters["recovery_jobs"] == 11
        assert counters["early_generated_tokens"] == 80
        assert counters["discarded_early_tokens"] == 55
        assert counters["recovery_generated_tokens"] == 44
        assert counters["draft_materialized_tokens"] == sum(
            eager_backend.metrics.query_tokens[purpose]
            for purpose in ("proposal", "commit", "eager")
        )
        assert not eager.works and eager_backend.report()["eager_gpu_live_work"] == 0
        assert sum(purpose == "setup" for purpose, _ in eager_worker.calls) == 16
    finally:
        serial_backend.shutdown()
        eager_backend.shutdown()
    assert serial_worker.closed and eager_worker.closed
    assert not serial_worker.memory and not eager_worker.memory
