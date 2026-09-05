"""CPU tests for independent bookkeeping, physical-slot, and capacity domains."""

from copy import deepcopy

import pytest
from test_phase4_dual_commit import EOS, RID, PhysicalRows, observer, row_context

from specrhythm.phase4.dual_rows import align_sampled_rows


def project(context, deltas, counts, tokens, materialized=None):
    return align_sampled_rows(
        context, deltas, counts, PhysicalRows(tokens),
        physical_request_ids=context["physical_request_ids"],
        target_materialized_token_counts=materialized,
    )


def test_one_sampled_request_from_100_physical_requests_with_256_capacity():
    physical_ids = [f"request-{index}" for index in range(100)]
    context = row_context([physical_ids[73]], physical_ids)
    rows = [(index, 900 + index) for index in range(256)]
    aligned = project(context, [[973]], [2] * 256, rows, list(range(100)))
    assert aligned.request_ids == ("request-73",)
    assert aligned.physical_indices == (73,)
    assert aligned.sampled_tokens == ((973,),)
    assert aligned.physical_tokens == ((73, 973),)
    assert aligned.materialized_counts == (73,)


def test_pinned_active_batch_one_row_and_capacity_100_are_distinct():
    aligned = project(row_context(), [[13]], [2] + [-99] * 99, [(1, 13)] + [()] * 99)
    assert aligned.request_ids == ("opaque",)
    assert aligned.physical_tokens == ((1, 13),)


def test_bookkeeping_order_survives_different_physical_compaction_and_discarded_rows():
    context = row_context(["c", "a"], ["b", "a", "unused", "c"])
    aligned = project(context, [[], [9]], [1, 2, 0, 2, 0], [(3,), (1, 9), (), (4, 5), ()])
    assert aligned.request_ids == ("c", "a")
    assert aligned.sampled_tokens == ((), (9,))
    assert aligned.physical_indices == (3, 1)
    assert aligned.physical_tokens == ((4, 5), (1, 9))


@pytest.mark.parametrize("mutation", ["duplicate", "sample-map-missing", "sample-map-index",
                                    "physical-map-missing", "physical-map-alias",
                                    "physical-id-missing", "sample-count", "capacity",
                                    "count-overflow", "schedule-mismatch", "spec-mismatch"])
def test_invalid_or_ambiguous_mapping_is_fatal(mutation):
    context = row_context(["a"], ["b", "a"])
    deltas, counts, tokens = [[9]], [1, 2], [(3,), (1, 9)]
    if mutation == "duplicate":
        context["sampled_request_ids"] = ["a", "a"]
    elif mutation == "sample-map-missing":
        context["req_id_to_sampled_index"] = {}
    elif mutation == "sample-map-index":
        context["req_id_to_sampled_index"]["a"] = 1
    elif mutation == "physical-map-missing":
        context["req_id_to_physical_index"].pop("a")
    elif mutation == "physical-map-alias":
        context["req_id_to_physical_index"]["a"] = 0
    elif mutation == "physical-id-missing":
        context["physical_request_ids"] = ["b"]
        context["req_id_to_physical_index"] = {"b": 0}
    elif mutation == "sample-count":
        deltas = []
    elif mutation == "capacity":
        counts = [1]
    elif mutation == "count-overflow":
        counts[1] = 3
    elif mutation == "schedule-mismatch":
        context["scheduled_request_ids"] = ["b"]
    elif mutation == "spec-mismatch":
        context["scheduled_spec_request_ids"] = ["b"]
    with pytest.raises(ValueError):
        project(context, deltas, counts, tokens)


@pytest.mark.parametrize("proposal,delta,maximum,expected", [
    ((13, EOS), (13, EOS, 151643), 8, (45596, 13, EOS)),
    ((13, 14), (13, 99), 8, (45596, 13, 99)),
    ((13, 14), (13, 14, 15), 4, (45596, 13, 14, 15)),
])
def test_proposer_uses_only_mapped_delta_preserving_eos_rejection_and_budget(
    tmp_path, proposal, delta, maximum, expected,
):
    obj, backend, _ = observer(tmp_path, proposal=proposal, maximum=maximum)
    physical_ids = [f"unscheduled-{index}" for index in range(100)]
    physical_ids[73] = "opaque"
    tokens = [()] * 256
    tokens[73] = obj.requests[RID].committed_token_ids + delta
    counts = [0] * 256
    counts[73] = len(tokens[73])
    context = row_context(["opaque"], physical_ids)
    context["scheduled_spec_request_ids"] = ["opaque"]
    result = obj.propose([list(delta)], counts, PhysicalRows(tokens), request_ids=physical_ids,
                         sampled_row_context=context)
    assert result == [[] for _ in physical_ids]  # Return remains in physical InputBatch domain.
    assert obj.requests[RID].generated_token_ids == expected
    if EOS in expected:
        assert len(obj.requests[RID].committed_token_ids) == 83
        assert obj.requests[RID].lifecycle == "TERMINAL"
        assert backend.calls.count(("propose",)) == 1


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("failure", ["signature", "peer-invalid", "physical-domain"])
def test_tp_row_contract_failure_is_seen_by_both_ranks_before_mutation(tmp_path, rank, failure):
    obj, _, _ = observer(tmp_path)
    obj.tp_rank = rank
    before = deepcopy(obj.requests[RID])

    def gather(rows, value, group):
        peer = (
            {**value, "signature": "wrong"} if failure == "signature" else
            {**value, "physical_request_ids": ["other"]} if failure == "physical-domain" else
            {"valid": False, "error": "missing mapping on peer"}
        )
        rows[:] = [value, peer] if rank == 0 else [peer, value]

    obj.dist.all_gather_object = gather
    physical = before.committed_token_ids + (13, EOS, 7)
    with pytest.raises(RuntimeError, match="TP sampled-row"):
        obj.propose([[13, EOS, 7]], [len(physical)], PhysicalRows([physical]),
                    request_ids=["opaque"], sampled_row_context=row_context())
    assert obj.requests[RID] == before
    assert obj.client.calls == []


def test_old_worker_hook_is_rejected_with_patch_requirement(tmp_path):
    obj, _, _ = observer(tmp_path)
    with pytest.raises(RuntimeError, match="sampled-row-context vLLM patch"):
        obj.propose([[13]], [82], PhysicalRows([tuple(range(80)) + (45596, 13)]),
                    request_ids=["opaque"])
