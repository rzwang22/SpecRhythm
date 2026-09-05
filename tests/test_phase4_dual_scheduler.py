"""Exercise the real Dual scheduler lifecycle with a CPU-only stock scheduler stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from specrhythm.phase4.dual import DualProposal, proposal_identity
from specrhythm.phase4.dual_correctness import (
    validate_proposal_lifecycle_events,
    validate_scheduler_cycles,
)
from specrhythm.phase4.dual_runner import summarize_retired_ready_results
from specrhythm.phase4.request_identity import (
    resolve_historical_ready_request,
    resolve_stable_ready_request,
)
from specrhythm.phase4.serial import token_prefix_hash


class Request:
    def __init__(self, internal_id, prompt):
        self.request_id = internal_id
        self.prompt_token_ids = list(prompt)
        self.all_token_ids = [*prompt, 10]
        self.num_prompt_tokens = len(prompt)
        self.num_computed_tokens = len(prompt)
        self.num_output_tokens = 1
        self.spec_token_ids = []
        self.finished = False

    def is_finished(self):
        return self.finished


class StockSchedulerStub:
    def __init__(self):
        self.requests = {
            "opaque-a": Request("opaque-a", (1, 2)),
            "opaque-b": Request("opaque-b", (3, 4)),
        }
        self.running = list(self.requests.values())
        self.current_step = 0
        self.force_schedule = False

    def schedule(self):
        eligible = [
            row for row in self.running
            if self.force_schedule or self._request_admissible_for_schedule(row)
        ]
        self.current_step += 1
        return SimpleNamespace(
            num_scheduled_tokens={row.request_id: 1 + len(row.spec_token_ids) for row in eligible},
            scheduled_spec_decode_tokens={
                row.request_id: list(row.spec_token_ids) for row in eligible if row.spec_token_ids
            },
        )


class DraftClientStub:
    def __init__(self):
        self.responses = []
        self.poll_limits = []

    def call(self, command, payload):
        assert command == "poll_ready"
        self.poll_limits.append(payload["limit"])
        if self.responses:
            return self.responses.pop(0)
        return {"ready": [], "pending_request_ids": []}


@pytest.fixture
def scheduler(monkeypatch, tmp_path):
    # Load only this module against a stub. No vLLM installation or GPU import,
    # and monkeypatch restores the import table for other Target/Serial tests.
    stub = ModuleType("vllm.v1.core.sched.scheduler")
    stub.Scheduler = StockSchedulerStub
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).resolve().parents[1] / "src/specrhythm/phase4/vllm_dual_scheduler.py"
    spec = importlib.util.spec_from_file_location("cpu_dual_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = DraftClientStub()
    monkeypatch.setattr(module, "DualDraftClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(module, "load_smoke_requests", lambda *args, **kwargs: [
        SimpleNamespace(request_id="a", prompt_token_ids=(1, 2)),
        SimpleNamespace(request_id="b", prompt_token_ids=(3, 4)),
    ])
    for key, value in {
        "SR_PHASE4_DUAL_BATCH": "1",
        "SR_PHASE4_DUAL_DRAFT_SOCKET": str(tmp_path / "draft.sock"),
        "SR_PHASE4_DUAL_SCHEDULER_EVENTS": str(tmp_path / "scheduler.jsonl"),
        "SR_PHASE4_PROPOSAL_LIFECYCLE_EVENTS": str(tmp_path / "lifecycle.jsonl"),
        "SR_PHASE4_DUAL_RESIDENT": "0",
        "SR_PHASE4_DUAL_MICROBATCH_SIZE": "1",
        "SR_PHASE4_WORKLOAD": str(tmp_path / "workload.jsonl"),
        "SR_PHASE4_REQUEST_COUNT": "2",
        "SR_PHASE4_DUAL_TEST_COORDINATION": "none",
    }.items():
        monkeypatch.setenv(key, value)
    instance = module.DualBatchScheduler()
    instance._bind_vllm_requests()
    return instance


def proposal_result(request_id="a", prefix=None, version=1, round_id=0, tokens=(11, 12)):
    prefix = prefix or ((1, 2, 10) if request_id == "a" else (3, 4, 10))
    proposal = DualProposal(
        request_id=request_id,
        round_id=round_id,
        proposal_id=proposal_identity(request_id, round_id, version, tokens),
        prefix_version=version,
        prefix_token_count=len(prefix),
        prefix_token_sha256=token_prefix_hash(prefix),
        draft_kv_length_before=len(prefix),
        draft_kv_length_after=len(prefix) + len(tokens),
        proposal_token_ids=tokens,
        created_timestamp_ns=30,
        draft_start_ns=10,
        draft_end_ns=20,
    )
    return {"request_id": request_id, "target_tail": False, "proposal": proposal.to_dict()}


def tail_result(request_id="a", timestamp=30):
    return {
        "request_id": request_id,
        "target_tail": True,
        "target_tail_ready_ns": timestamp,
        "proposal": None,
    }


def retire(scheduler, stable_id="a"):
    internal_id = scheduler._dual_identity.internal_id(stable_id)
    request = scheduler.requests.pop(internal_id)
    scheduler.running.remove(request)
    return request


def lifecycle(scheduler):
    return scheduler._dual_proposal_lifecycle.read()


@pytest.mark.parametrize("stable_id", ["never-known", "a"])
def test_unknown_or_never_bound_ready_result_is_fatal(scheduler, stable_id):
    scheduler._dual_identity.stable_to_internal.pop("a")
    scheduler._dual_identity.internal_to_stable.pop("opaque-a")
    with pytest.raises(RuntimeError, match="no live vLLM identity binding"):
        scheduler._accept_ready_result(proposal_result(stable_id))
    assert lifecycle(scheduler) == []


@pytest.mark.parametrize("stable_id", [None, "", 123, "   "])
def test_invalid_ready_identity_is_fatal(scheduler, stable_id):
    result = tail_result()
    result["request_id"] = stable_id
    with pytest.raises(RuntimeError, match="non-empty string"):
        scheduler._accept_ready_result(result)


def test_live_proposal_installs_only_on_its_owner(scheduler):
    scheduler._dual_drafting.add("a")
    scheduler._accept_ready_result(proposal_result())
    assert scheduler.requests["opaque-a"].spec_token_ids == [11, 12]
    assert scheduler.requests["opaque-b"].spec_token_ids == []
    assert "a" not in scheduler._dual_drafting
    assert [row["lifecycle_state"] for row in lifecycle(scheduler)] == [
        "CREATED", "PUBLISHED", "INSTALLED",
    ]
    assert scheduler._dual_retired_ready_events == []


def test_live_tail_remains_admissible(scheduler):
    scheduler._accept_ready_result(tail_result())
    assert scheduler._dual_tail_ready == {"a"}
    assert scheduler._dual_tail_ready_ns == {"a": 30}
    assert scheduler.schedule().num_scheduled_tokens == {"opaque-a": 1}


@pytest.mark.parametrize("result_kind", ["proposal", "tail"])
@pytest.mark.parametrize("still_in_table", [False, True])
def test_retired_or_terminal_result_is_validated_dropped_and_cleared(
    scheduler, result_kind, still_in_table,
):
    request = scheduler.requests["opaque-a"] if still_in_table else retire(scheduler)
    request.finished = True
    scheduler._dual_drafting.update(("a", "b"))
    scheduler._dual_tail_ready.update(("a", "b"))
    scheduler._dual_tail_ready_ns.update(a=1, b=2)
    result = proposal_result() if result_kind == "proposal" else tail_result()
    scheduler._accept_ready_result(result)
    assert request.spec_token_ids == []  # Even the detached object is untouched.
    assert scheduler.requests["opaque-b"].spec_token_ids == []
    assert scheduler._dual_drafting == {"b"}
    assert scheduler._dual_tail_ready == {"b"}
    assert scheduler._dual_tail_ready_ns == {"b": 2}
    assert "a" not in scheduler._dual_proposals
    events = scheduler._dual_retired_ready_events
    assert len(events) == 1
    assert events[0]["reason"] == (
        "terminal-request" if still_in_table else "request-retired-before-ready"
    )
    assert events[0]["discarded"] and not events[0]["installed"] and not events[0]["verified"]
    states = [row["lifecycle_state"] for row in lifecycle(scheduler)]
    expected = ["CREATED", "PUBLISHED", "DROPPED_STALE"] if result_kind == "proposal" else []
    assert states == expected
    if states:
        assert validate_proposal_lifecycle_events(lifecycle(scheduler)) == []
    # Delivery replay must produce neither another drop nor duplicate publication.
    scheduler._accept_ready_result(result)
    assert scheduler._dual_retired_ready_events == events
    assert len(events) == 1
    assert [row["lifecycle_state"] for row in lifecycle(scheduler)] == states


def test_strict_live_resolver_unchanged_for_removed_request(scheduler):
    retire(scheduler)
    identity = scheduler._dual_identity
    resolved = resolve_historical_ready_request("a", identity, scheduler.requests)
    assert resolved == ("opaque-a", None)
    with pytest.raises(RuntimeError, match="has no mapped vLLM request"):
        resolve_stable_ready_request("a", identity, scheduler.requests)


@pytest.mark.parametrize("corruption", ["reverse", "alias-stable", "alias-internal", "frozen"])
def test_retired_identity_corruption_is_fatal(scheduler, corruption):
    retire(scheduler)
    identity = scheduler._dual_identity
    if corruption == "reverse":
        identity.internal_to_stable["opaque-a"] = "b"
    elif corruption == "alias-stable":
        identity.stable_to_internal["b"] = "opaque-a"
    elif corruption == "alias-internal":
        identity.internal_to_stable["alias"] = "a"
    else:
        identity.stable_prompts.pop("a")
    with pytest.raises(RuntimeError, match="binding is inconsistent"):
        scheduler._accept_ready_result(proposal_result())
    assert scheduler._dual_retired_ready_events == []


@pytest.mark.parametrize("corruption", ["object-id", "null-entry"])
def test_live_request_table_corruption_is_fatal(scheduler, corruption):
    if corruption == "object-id":
        scheduler.requests["opaque-a"].request_id = "opaque-b"
    else:
        scheduler.requests["opaque-a"] = None
    with pytest.raises(RuntimeError):
        scheduler._accept_ready_result(proposal_result())
    assert scheduler._dual_retired_ready_events == []


@pytest.mark.parametrize("field,value", [
    ("proposal_id", "wrong-canonical-id"),
    ("protocol_version", "bad-protocol"),
    ("proposal_token_ids", []),
    ("draft_kv_length_after", 900),
    ("prefix_version", -1),
    ("round_id", True),
    ("request_id", None),
    ("prefix_token_sha256", None),
    ("prefix_token_sha256", "not-a-sha"),
    ("proposal_length", 3),
    ("proposal_length", True),
])
def test_malformed_retired_proposal_is_fatal(scheduler, field, value):
    retire(scheduler)
    result = proposal_result()
    result["proposal"][field] = value
    with pytest.raises(ValueError):
        scheduler._accept_ready_result(result)
    assert lifecycle(scheduler) == []
    assert scheduler._dual_retired_ready_events == []


@pytest.mark.parametrize("value", [[], "tokens", 123])
def test_retired_nonobject_proposal_is_fatal(scheduler, value):
    retire(scheduler)
    with pytest.raises(RuntimeError, match="payload is not an object"):
        scheduler._accept_ready_result({"request_id": "a", "proposal": value})


def test_retired_wrong_proposal_request_id_is_fatal(scheduler):
    retire(scheduler)
    result = proposal_result("b")
    result["request_id"] = "a"
    with pytest.raises(RuntimeError, match="request_id disagrees"):
        scheduler._accept_ready_result(result)
    assert lifecycle(scheduler) == []


@pytest.mark.parametrize("field,value", [
    ("target_tail", False), ("target_tail", "true"),
    ("target_tail_ready_ns", None), ("target_tail_ready_ns", 0),
    ("target_tail_ready_ns", True), ("target_tail_ready_ns", "30"),
    ("terminal", True),
])
def test_malformed_retired_tail_is_fatal(scheduler, field, value):
    retire(scheduler)
    result = tail_result()
    result[field] = value
    with pytest.raises(RuntimeError):
        scheduler._accept_ready_result(result)
    assert scheduler._dual_tail_ready == set()
    assert scheduler._dual_retired_ready_events == []


def test_retired_proposal_and_tail_contradiction_is_fatal(scheduler):
    retire(scheduler)
    result = proposal_result()
    result["target_tail"] = True
    with pytest.raises(RuntimeError, match="also claims a target tail"):
        scheduler._accept_ready_result(result)


@pytest.mark.parametrize("kwargs,reason", [
    ({"prefix": (1, 2, 10, 13)}, "prefix_token_count"),
    ({"prefix": (1, 2, 99)}, "prefix_token_sha256"),
    ({"version": 2}, "prefix_version"),
    ({"round_id": 1}, "round_id"),
])
def test_live_proposal_guards_remain_fatal(scheduler, kwargs, reason):
    with pytest.raises(RuntimeError, match=reason):
        scheduler._accept_ready_result(proposal_result(**kwargs))
    assert scheduler.requests["opaque-a"].spec_token_ids == []


def test_second_unverified_live_proposal_remains_fatal(scheduler):
    scheduler._accept_ready_result(proposal_result())
    with pytest.raises(RuntimeError, match="second_unverified_proposal"):
        scheduler._accept_ready_result(proposal_result(version=2, round_id=1))
    assert scheduler.requests["opaque-a"].spec_token_ids == [11, 12]


@pytest.mark.parametrize("removed", [False, True])
def test_consumed_proposal_redelivery_remains_fatal(scheduler, removed):
    result = proposal_result()
    scheduler._accept_ready_result(result)
    scheduler.schedule()
    if removed:
        retire(scheduler)
    with pytest.raises(RuntimeError, match="consumed more than once"):
        scheduler._accept_ready_result(result)
    assert validate_proposal_lifecycle_events(lifecycle(scheduler)) == []


def test_duplicate_stock_consumption_remains_fatal(scheduler):
    scheduler._accept_ready_result(proposal_result())
    scheduler.schedule()
    scheduler.force_schedule = True
    with pytest.raises(RuntimeError, match="consumed more than once"):
        scheduler.schedule()


def test_consumed_history_survives_retired_cleanup(scheduler):
    scheduler._accept_ready_result(proposal_result())
    scheduler.schedule()
    consumed_id = scheduler._dual_proposals["a"].proposal_id
    retire(scheduler)
    scheduler._accept_ready_result(proposal_result(version=2, round_id=1))
    assert "a" not in scheduler._dual_proposals
    assert consumed_id in scheduler._dual_consumed_proposals
    rows = lifecycle(scheduler)
    assert [row["lifecycle_state"] for row in rows if row["proposal_id"] == consumed_id] == [
        "CREATED", "PUBLISHED", "INSTALLED", "CONSUMED",
    ]
    assert validate_proposal_lifecycle_events(rows) == []


@pytest.mark.parametrize("tail_first", [False, True])
def test_installed_unconsumed_cleanup_has_one_drop_and_no_republication(scheduler, tail_first):
    result = proposal_result()
    scheduler._accept_ready_result(result)
    retire(scheduler)
    if tail_first:
        scheduler._accept_ready_result(tail_result())
    scheduler._accept_ready_result(result)
    scheduler._accept_ready_result(result)
    assert [row["lifecycle_state"] for row in lifecycle(scheduler)] == [
        "CREATED", "PUBLISHED", "INSTALLED", "DROPPED_STALE",
    ]
    assert validate_proposal_lifecycle_events(lifecycle(scheduler)) == []
    assert "a" not in scheduler._dual_proposals


def test_retired_replay_changed_payload_is_fatal(scheduler):
    retire(scheduler)
    result = proposal_result()
    scheduler._accept_ready_result(result)
    result["proposal"]["prefix_token_sha256"] = token_prefix_hash((1, 2, 99))
    with pytest.raises(RuntimeError, match="changed its payload"):
        scheduler._accept_ready_result(result)
    assert len(lifecycle(scheduler)) == 3


def test_retired_conflicting_proposal_for_same_round_is_fatal(scheduler):
    retire(scheduler)
    scheduler._accept_ready_result(proposal_result())
    with pytest.raises(RuntimeError, match="one request round"):
        scheduler._accept_ready_result(proposal_result(tokens=(13, 14)))
    assert len(lifecycle(scheduler)) == 3


@pytest.mark.parametrize("late_result", [proposal_result, tail_result])
@pytest.mark.parametrize("same_poll", [False, True])
def test_retired_result_releases_capacity_and_other_request_progresses(
    scheduler, late_result, same_poll,
):
    retire(scheduler)
    scheduler._dual_drafting.add("a")
    scheduler._dual_microbatch_size = 2 if same_poll else 1
    scheduler._dual_client.responses = [
        {"ready": [late_result(), *([proposal_result("b")] if same_poll else [])],
         "pending_request_ids": ["a"]},
        {"ready": [] if same_poll else [proposal_result("b")], "pending_request_ids": []},
    ]
    first = scheduler.schedule()
    assert "a" not in scheduler._dual_drafting
    assert "a" not in scheduler._dual_tail_ready
    assert "a" not in scheduler._dual_proposals
    assert "opaque-a" not in scheduler.requests
    if same_poll:
        assert first.scheduled_spec_decode_tokens == {"opaque-b": [11, 12]}
        scheduler.requests["opaque-b"].spec_token_ids = []
    else:
        assert first.num_scheduled_tokens == {}
    second = scheduler.schedule()
    if not same_poll:
        assert second.scheduled_spec_decode_tokens == {"opaque-b": [11, 12]}
    assert scheduler._dual_client.poll_limits == [scheduler._dual_microbatch_size] * 2
    rows = scheduler._dual_events.read()
    assert len(rows[0]["retired_ready_results"]) == 1
    assert rows[1]["retired_ready_results"] == []
    assert validate_proposal_lifecycle_events(lifecycle(scheduler)) == []
    assert validate_scheduler_cycles(rows, proposal_lifecycle_rows=lifecycle(scheduler)) == []
    summary = summarize_retired_ready_results(rows)
    assert summary["retired_ready_result_drop_count"] == 1
    assert summary["retired_proposal_drop_count"] == int(late_result is proposal_result)
    assert summary["retired_tail_drop_count"] == int(late_result is tail_result)


def test_final_artifact_retired_summary_explicitly_distinguishes_zero_and_nonzero(scheduler):
    assert summarize_retired_ready_results([]) == {
        "retired_ready_result_drop_count": 0,
        "retired_proposal_drop_count": 0,
        "retired_tail_drop_count": 0,
        "events": [],
    }
    retire(scheduler)
    scheduler._accept_ready_result(proposal_result())
    scheduler._accept_ready_result(tail_result())
    scheduler.schedule()
    summary = summarize_retired_ready_results(scheduler._dual_events.read())
    assert summary["retired_ready_result_drop_count"] == 2
    assert summary["retired_proposal_drop_count"] == summary["retired_tail_drop_count"] == 1
    assert len(summary["events"]) == 2
