"""CPU runtime regression: rejection-parsed deltas precede serving stop truncation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from specrhythm.phase4.draft_service import HFPersistentDraftBackend, _HFRequest
from specrhythm.phase4.dual import DualProposal
from specrhythm.phase4.dual_commit import DualStopPolicy, load_dual_stop_policies
from specrhythm.phase4.dual_correctness import (
    validate_request_state_events,
    validate_round_accounting,
)
from specrhythm.phase4.dual_rows import ROW_CONTEXT_SCHEMA
from specrhythm.phase4.dual_service import AsyncDualDraftController, DualDraftMachine
from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_dual import DualBatchRemoteProposer, _Request

RID = "r3-22887f929fd54d97814c2bd3"
EOS = 151645


class PhysicalRows:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        index, section = key
        return SimpleNamespace(tolist=lambda: list(self.rows[index][section]))

    def __len__(self):
        return len(self.rows)


def row_context(ids=("opaque",), physical_ids=None):
    physical_ids = ids if physical_ids is None else physical_ids
    return {
        "schema_version": ROW_CONTEXT_SCHEMA,
        "sampled_request_ids": list(ids),
        "req_id_to_sampled_index": {key: index for index, key in enumerate(ids)},
        "physical_request_ids": list(physical_ids),
        "req_id_to_physical_index": {key: index for index, key in enumerate(physical_ids)},
        "scheduled_request_ids": list(ids), "scheduled_spec_request_ids": [],
    }


class Backend:
    provenance = {"physical_gpu_id": 0}

    def __init__(self, proposal):
        self.proposal = tuple(proposal)
        self.prefix = ()
        self.calls = []

    def initialize(self, request_id, prefix):
        self.prefix = tuple(prefix)

    def propose(self, request_id, budget, eos_token_ids):
        self.calls.append(("propose",))
        return self.proposal[:budget], len(self.proposal[:budget])

    def rollback(self, request_id, accepted):
        self.calls.append(("rollback", accepted))
        self.prefix += self.proposal[:accepted]

    def append_target_token(self, request_id, token):
        self.calls.append(("append", token))
        self.prefix += (token,)

    def finish(self, request_id):
        self.calls.append(("finish",))

    def shutdown(self):
        pass


class Client:
    def __init__(self, machine, ready):
        self.machine, self.ready = machine, ready
        self.calls, self.results = [], []

    def call(self, command, payload):
        self.calls.append((command, payload))
        if command == "claimed":
            return {"claimed": [self.ready]}
        assert command == "enqueue"
        for row in payload["rows"]:
            self.results.append(getattr(self.machine, payload["work_operation"])(row))
        return {"blocking_on_draft_gpu": False}


def observer(tmp_path, *, proposal=(13, EOS), maximum=8, eos=EOS, stops=(), tail=False):
    definition = SimpleNamespace(
        request_id=RID, prompt_token_ids=tuple(range(80)),
        maximum_new_tokens=maximum, sampling_seed=42,
    )
    obj = DualBatchRemoteProposer.__new__(DualBatchRemoteProposer)
    obj.definitions = {RID: definition}
    obj.identity = FrozenPromptIdentityMap.from_definitions([definition])
    obj.identity.bind("opaque", definition.prompt_token_ids)
    obj.stable_to_internal = obj.identity.stable_to_internal
    obj.stop_policies = {RID: DualStopPolicy(maximum, eos, stops)}
    prefix = definition.prompt_token_ids + (45596,)
    obj.requests = {RID: _Request(definition.prompt_token_ids, maximum, prefix, (45596,))}
    obj.state_log = CheckpointJsonl(tmp_path / "states.jsonl")
    obj.proposal_log = CheckpointJsonl(tmp_path / "proposals.jsonl")
    obj.timing_log = CheckpointJsonl(tmp_path / "timing.jsonl")
    obj._write_report = lambda: None
    obj.resident_decode_ready, obj.setup_complete, obj.tp_rank = True, True, 0
    obj.tp_world_size = 2
    obj.dist = SimpleNamespace(all_gather_object=lambda rows, value, group:
                               rows.__setitem__(slice(None), [value, value]))
    obj.tp_group = SimpleNamespace(broadcast_object=lambda result, src: result, cpu_group=None)
    backend = Backend(proposal)
    machine = DualDraftMachine(backend)
    obj._transition(RID, "DRAFT_READY")
    obj._transition(RID, "DRAFTING")
    ready = machine.bootstrap_and_propose(obj._work_row(RID, obj.requests[RID], terminal=False))
    obj.client = Client(machine, ready)
    if tail:
        assert ready["target_tail"] is True
        obj._verified_ids, obj._verify_batch_by_request = set(), {}
    else:
        obj.requests[RID].pending_proposal = DualProposal.from_dict(ready["proposal"])
        for destination in ("PROPOSAL_READY", "VERIFY_READY", "VERIFYING"):
            obj._transition(RID, destination)
        obj._verified_ids = {RID}
        obj._verify_batch_by_request = {RID: "verify-0"}
    return obj, backend, machine


def update(obj, sampled, physical=None):
    if physical is None:
        physical = obj.requests[RID].committed_token_ids + tuple(sampled)
    assert obj.propose([list(sampled)], [len(physical)], PhysicalRows([physical]),
                       request_ids=["opaque"], sampled_row_context=row_context()) == [[]]


@pytest.mark.parametrize("proposal,sampled,maximum,expected,reason,accepted,correction,bonus", [
    ((13, 14), (99,), 8, (99,), None, (), (99,), ()),
    ((13, 14), (13, 14, 15), 8, (13, 14, 15), None, (13, 14), (), (15,)),
    ((13, 14, 15), (13, 99), 8, (13, 99), None, (13,), (99,), ()),
    ((13, 14), (13, EOS), 8, (13, EOS), "eos", (13,), (EOS,), ()),
    ((13, 14), (13, 14, EOS), 8, (13, 14, EOS), "eos", (13, 14), (), (EOS,)),
    ((13, EOS), (13, EOS, 7), 8, (13, EOS), "eos", (13, EOS), (), ()),
    ((EOS, 13), (EOS, 13, 7), 8, (EOS,), "eos", (EOS,), (), ()),
    ((13, 14, 15), (EOS,), 8, (EOS,), "eos", (), (EOS,), ()),
    # Terminal truncation within an accepted Draft prefix: no correction/bonus.
    ((13, EOS, 15), (13, EOS, 15, 7), 8, (13, EOS), "eos", (13, EOS), (), ()),
    ((13, 14, 15, 16), (13, 14, 15, 16, 7), 6,
     (13, 14, 15, 16, 7), "max_tokens", (13, 14, 15, 16), (), (7,)),
])
def test_observer_commits_serving_delta_and_synchronizes_draft(
    tmp_path, proposal, sampled, maximum, expected, reason, accepted, correction, bonus,
):
    obj, backend, machine = observer(tmp_path, proposal=proposal, maximum=maximum)
    before = obj.requests[RID].committed_token_ids
    physical = before + sampled
    update(obj, sampled, physical)
    state = obj.requests[RID]
    assert state.generated_token_ids == (45596,) + expected
    assert state.committed_token_ids == before + expected
    assert physical[:len(state.committed_token_ids)] == state.committed_token_ids
    assert state.prefix_version == 2 and state.next_round_id == 1
    assert state.pending_proposal is None
    assert state.terminal is (reason is not None)
    row, = obj.proposal_log.read()
    assert row["round_id"] == 0 and row["prefix_version"] == 1
    assert row["committed_token_ids"] == list(expected)
    assert row["accepted_draft_token_ids"] == list(accepted)
    assert row["rejected_draft_token_ids"] == list(proposal[len(accepted):])
    assert row["target_correction_token_ids"] == list(correction)
    assert row["target_bonus_token_ids"] == list(bonus)
    assert row["terminal_truncation_reason"] == reason
    assert validate_round_accounting([row]) == []
    assert machine.requests[RID].committed_token_ids == state.committed_token_ids
    assert backend.prefix == state.committed_token_ids
    assert ("rollback", len(accepted)) in backend.calls
    if correction or bonus:
        assert ("append", (correction or bonus)[0]) in backend.calls
    else:
        assert not any(call[0] == "append" for call in backend.calls)
    sync = obj.client.results[-1]
    assert sync["logical_draft_kv_length"] == len(state.committed_token_ids)
    assert sync["prefix_token_sha256"] == token_prefix_hash(state.committed_token_ids)
    if reason:
        assert sync["proposal"] is None and sync["target_tail"] is False
        assert machine.requests[RID].finished is True
        assert backend.calls.count(("propose",)) == 1
        assert backend.calls[-1] == ("finish",)
        assert state.lifecycle == "TERMINAL"
        events = obj.state_log.read()
        assert [row["destination_state"] for row in events[-2:]] == ["COMMITTING", "TERMINAL"]
        assert validate_request_state_events(events) == []
        snapshot = (len(events), len(obj.client.calls), state.prefix_version, state.next_round_id)
        update(obj, (), physical)  # Empty later callback cannot re-commit physical suffix.
        assert snapshot == (len(obj.state_log.read()), len(obj.client.calls),
                            state.prefix_version, state.next_round_id)
        with pytest.raises(RuntimeError, match="already terminal"):
            update(obj, (8,), physical + (8,))
    else:
        assert state.lifecycle == "DRAFT_SYNC"
        assert sync["proposal"]["round_id"] == 1
        assert sync["draft_sync_complete_ns"] <= sync["proposal"]["draft_start_ns"]


def test_real_80_plus_3_regression_has_no_round_one_proposal(tmp_path):
    obj, backend, machine = observer(tmp_path)
    physical = tuple(range(80)) + (45596, 13, EOS, 151643)
    assert len(physical) == 84
    update(obj, (13, EOS, 151643), physical)
    state = obj.requests[RID]
    assert state.generated_token_ids == (45596, 13, EOS)
    assert len(state.committed_token_ids) == 83
    assert state.lifecycle == "TERMINAL"
    assert all(event["committed_prefix_length"] <= 83 for event in obj.state_log.read())
    assert [row["round_id"] for row in obj.proposal_log.read()] == [0]
    assert machine.requests[RID].proposal is None
    assert backend.calls.count(("propose",)) == 1


@pytest.mark.parametrize("sampled", [(13, 14, 15, 16, 17), (13, 14, 15, EOS, 17)])
def test_remaining_budget_trims_a_long_round_without_post_budget_commit(tmp_path, sampled):
    obj, backend, machine = observer(tmp_path, proposal=(13, 14, 15, 16), maximum=8)
    # Exercise the commit boundary independently of the ordinary remaining-1
    # Draft budget cap, retaining the full rejection-parsed physical evidence.
    obj.requests[RID].maximum_new_tokens = 3
    obj.stop_policies[RID] = DualStopPolicy(3, EOS)
    # Second case must also be a valid rejection (correction at proposal index 3).
    if EOS in sampled:
        sampled = sampled[:-1]
    update(obj, sampled)
    assert obj.requests[RID].generated_token_ids == (45596, 13, 14)
    assert obj.requests[RID].lifecycle == "TERMINAL"
    assert machine.requests[RID].committed_token_ids == backend.prefix
    assert backend.calls == [("propose",), ("rollback", 2), ("finish",)]
    assert obj.proposal_log.read()[0]["terminal_truncation_reason"] == "max_tokens"


@pytest.mark.parametrize("token", [13, EOS])
def test_proposal_free_terminal_target_tail(tmp_path, token):
    obj, backend, machine = observer(tmp_path, maximum=2, tail=True)
    update(obj, (token,))
    assert obj.requests[RID].generated_token_ids == (45596, token)
    assert obj.requests[RID].prefix_version == 2
    assert obj.requests[RID].next_round_id == 0
    assert obj.requests[RID].lifecycle == "TERMINAL"
    assert obj.proposal_log.read() == []
    assert validate_request_state_events(obj.state_log.read()) == []
    assert backend.calls == [("append", token), ("finish",)]
    assert machine.requests[RID].finished is True


@pytest.mark.parametrize("change", ["extra-physical", "missing-physical", "diverged-prior",
                                    "invalid-token", "wrong-delta", "empty-verified"])
def test_physical_rows_never_override_delta_or_rejection_evidence(tmp_path, change):
    obj, _, _ = observer(tmp_path)
    previous = obj.requests[RID].committed_token_ids
    sampled = (13, EOS, 7)
    physical = previous + sampled
    if change == "extra-physical":
        physical += (8,)
    elif change == "missing-physical":
        physical = physical[:-1]
    elif change == "diverged-prior":
        physical = physical[:80] + (1,) + physical[81:]
    elif change == "invalid-token":
        sampled, physical = (-1,), previous + (-1,)
    elif change == "wrong-delta":
        sampled, physical = (99, 100), previous + (99, 100)
    elif change == "empty-verified":
        sampled, physical = (), previous
    with pytest.raises((RuntimeError, ValueError)):
        update(obj, sampled, physical)
    assert obj.requests[RID].committed_token_ids == previous
    assert obj.proposal_log.read() == []
    assert obj.client.calls == []


def test_explicit_processed_stop_token_is_inclusive_and_distinct_from_eos(tmp_path):
    obj, _, _ = observer(tmp_path, proposal=(13, 77), stops=(77,))
    update(obj, (13, 77, 7))
    assert obj.requests[RID].generated_token_ids == (45596, 13, 77)
    assert obj.requests[RID].terminal is True
    assert obj.proposal_log.read()[0]["terminal_truncation_reason"] == "stop"


@pytest.mark.parametrize("rows,ids,counts", [([], ["opaque"], [81]),
                                          ([[], []], ["opaque", "opaque"], [81, 81]),
                                          ([[]], ["opaque"], []),
                                          ([], [], [])])
def test_missing_or_duplicate_verification_rows_fail_closed(tmp_path, rows, ids, counts):
    obj, _, _ = observer(tmp_path)
    with pytest.raises(RuntimeError, match="contract failed|missing sampled-token rows"):
        obj.propose(rows, counts, PhysicalRows([]), request_ids=ids,
                    sampled_row_context=row_context(ids))
    assert obj.requests[RID].prefix_version == 1
    assert obj.client.calls == []


def test_nonterminal_accepted_prefix_without_correction_remains_invalid(tmp_path):
    obj, _, _ = observer(tmp_path, proposal=(13, 14, 15))
    with pytest.raises(ValueError, match="exactly one Target correction"):
        update(obj, (13, 14))
    assert obj.requests[RID].prefix_version == 1
    assert obj.client.calls == []


def params(**changes):
    return SimpleNamespace(
        **dict(dict(stop=[], min_tokens=0, repetition_detection=None, temperature=0.0,
                    n=1, max_tokens=8, eos_token_id=EOS, stop_token_ids=[99],
                    ignore_eos=False), **changes)
    )


@pytest.mark.parametrize("changes", [dict(stop=["end"]), dict(min_tokens=1),
                                    dict(repetition_detection={}), dict(temperature=1),
                                    dict(n=2), dict(max_tokens=9),
                                    dict(stop_token_ids=[True]), dict(eos_token_id=-1),
                                    dict(ignore_eos=True)])
def test_unsupported_sampling_contract_fails_closed(changes):
    with pytest.raises(ValueError):
        DualStopPolicy.from_sampling_params(params(**changes), maximum=8,
                                            prompt_length=80, max_model_len=128)


def test_stop_priority_and_ignore_eos_contract():
    policy = DualStopPolicy.from_sampling_params(params(ignore_eos=True, eos_token_id=None),
                                                maximum=8, prompt_length=80, max_model_len=88)
    assert policy.canonicalize((1,), (EOS, 13)) == ((1, EOS, 13), None)
    assert policy.canonicalize((1,), (99, 13)) == ((1, 99), "stop")
    assert DualStopPolicy(2, EOS).canonicalize((1,), (EOS, 13)) == ((1, EOS), "eos")
    with pytest.raises(ValueError, match="context limit"):
        DualStopPolicy.from_sampling_params(params(), maximum=8,
                                            prompt_length=80, max_model_len=87)


def test_policy_uses_pinned_input_processor_sources_and_sampling_api(monkeypatch):
    calls = []

    class SamplingParams:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.__dict__.update(params().__dict__)
            self.__dict__.update(kwargs)

        def update_from_generation_config(self, generation, eos):
            assert generation == {"eos_token_id": [EOS, 99]}
            self.eos_token_id = eos
            self.stop_token_ids = [99]

        def update_from_tokenizer(self, tokenizer):
            assert tokenizer == "authoritative-tokenizer"

    vllm, renderers = ModuleType("vllm"), ModuleType("vllm.renderers")
    vllm.SamplingParams = SamplingParams
    renderers.renderer_from_config = lambda config: SimpleNamespace(
        tokenizer="authoritative-tokenizer", get_eos_token_id=lambda: EOS,
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.renderers", renderers)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(
            max_model_len=128, hf_config=SimpleNamespace(eos_token_id=123),
            try_get_generation_config=lambda: {"eos_token_id": [EOS, 99]},
        ),
    )
    definition = SimpleNamespace(request_id=RID, prompt_token_ids=tuple(range(80)),
                                 maximum_new_tokens=8, sampling_seed=42)
    policies = load_dual_stop_policies(config, [definition])
    assert policies[RID] == DualStopPolicy(8, EOS, (99,))
    assert calls == [dict(temperature=0.0, top_p=1.0, max_tokens=8, seed=42, n=1, logprobs=None)]
    config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="synchronous"):
        load_dual_stop_policies(config, [definition])


def test_async_final_draft_sync_does_not_publish_ready_work(tmp_path):
    obj, backend, machine = observer(tmp_path)
    prefix = obj.requests[RID].committed_token_ids + (13, EOS)
    proposal = obj.requests[RID].pending_proposal
    controller = AsyncDualDraftController(machine, CheckpointJsonl(tmp_path / "work.jsonl"))
    controller.enqueue("commit_and_propose", [{
        "request_id": RID, "proposal_id": proposal.proposal_id, "round_id": 0,
        "committed_delta": [13, EOS], "prefix_version": 2,
        "prefix_token_sha256": token_prefix_hash(prefix), "terminal": True,
        "remaining_output_budget": 5, "eos_token_ids": [EOS],
    }])
    status = controller.shutdown()  # Wait for CPU work, including final bookkeeping.
    assert status["failures"] == {}
    assert status["inflight_request_ids"] == []
    assert status["ready_request_ids"] == []
    assert status["claimed_request_ids"] == []
    assert backend.calls == [("propose",), ("rollback", 2), ("finish",)]
    event, = CheckpointJsonl(tmp_path / "work.jsonl").read()
    assert event["success"] is True
    assert event["result"]["logical_draft_kv_length"] == 83
    assert event["result"]["proposal"] is None


@pytest.mark.parametrize("accepted", [1, 2])
def test_persistent_backend_materializes_accepted_terminal_prefix_before_release(accepted):
    # Exercise real HF rollback bookkeeping with CPU cache/logit stand-ins.
    # The final proposed token has not yet been materialized when all accept.
    cache = SimpleNamespace(length=82)
    cache.crop = lambda count: setattr(cache, "length", min(cache.length, count))
    state = _HFRequest(cache, "after-13", 81, "bootstrap", (13, EOS),
                       ["bootstrap", "after-13"])
    backend = HFPersistentDraftBackend.__new__(HFPersistentDraftBackend)
    backend.states = {RID: state}
    appended = []

    def append(current, token):
        current.cache.length += 1
        appended.append(token)
        return "after-EOS"

    backend._append_token = append
    backend.rollback(RID, accepted)
    assert state.committed_length == cache.length == 81 + accepted
    assert appended == ([] if accepted == 1 else [EOS])
    backend.finish(RID)
    assert RID not in backend.states
