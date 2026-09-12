from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrhythm.phase4.decode_ready import (
    DecodeReadyRequest,
    build_first_target_forward_contract,
    validate_first_target_forward_contract,
)
from specrhythm.phase4.dual import DualProposal, proposal_identity
from specrhythm.phase4.dual_correctness import validate_verification_contracts
from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
from specrhythm.phase4.transport import CheckpointJsonl
from specrhythm.phase4.vllm_diagnostics import (
    capture_target_forward,
    diagnostic_proposal_id,
    validate_target_diagnostic,
)

REQUEST_ID = "r3-code-0"
COMMITTED_PREFIX = (1, 2, 3, 7)
PROPOSAL_TOKENS = (10, 11)


class _FakeTensor:
    def __init__(self, value):
        self.value = value
        self.dtype = "float32"

    @property
    def shape(self):
        if not isinstance(self.value, list):
            return ()
        if self.value and isinstance(self.value[0], list):
            return (len(self.value), len(self.value[0]))
        return (len(self.value),)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def item(self):
        return self.value

    def log_softmax(self, dim):
        assert dim == -1
        return self

    def topk(self, count):
        indexed = sorted(
            enumerate(self.value), key=lambda item: item[1], reverse=True
        )[:count]
        return (
            _FakeTensor([value for _index, value in indexed]),
            _FakeTensor([index for index, _value in indexed]),
        )

    def argmax(self):
        return _FakeTensor(max(range(len(self.value)), key=self.value.__getitem__))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            value = self.value[row][column]
        else:
            value = self.value[key]
        return _FakeTensor(value) if isinstance(value, list) else value


def _serial_proposal() -> Proposal:
    return Proposal(
        protocol_version=PROTOCOL_VERSION,
        request_id=REQUEST_ID,
        round_id=0,
        parent_prefix_len=len(COMMITTED_PREFIX),
        parent_prefix_hash=token_prefix_hash(COMMITTED_PREFIX),
        proposal_token_ids=PROPOSAL_TOKENS,
        proposal_eos=False,
        draft_start_ns=10,
        draft_end_ns=20,
        transport_payload_bytes=16,
        model_provenance={},
        runtime_provenance={},
    )


def _dual_proposal() -> DualProposal:
    identity = proposal_identity(REQUEST_ID, 0, 1, PROPOSAL_TOKENS)
    return DualProposal(
        request_id=REQUEST_ID,
        round_id=0,
        proposal_id=identity,
        prefix_version=1,
        prefix_token_count=len(COMMITTED_PREFIX),
        prefix_token_sha256=token_prefix_hash(COMMITTED_PREFIX),
        draft_kv_length_before=len(COMMITTED_PREFIX),
        draft_kv_length_after=len(COMMITTED_PREFIX) + len(PROPOSAL_TOKENS),
        proposal_token_ids=PROPOSAL_TOKENS,
        created_timestamp_ns=9,
        draft_start_ns=10,
        draft_end_ns=20,
    )


def _install_rank_zero_vllm(monkeypatch) -> None:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)


def _capture(tmp_path, monkeypatch, pending):
    _install_rank_zero_vllm(monkeypatch)
    fixture = Path(__file__).parent / "fixtures" / "phase4-r3-smoke.jsonl"
    output = tmp_path / "target-diagnostics.jsonl"
    monkeypatch.setenv("SR_PHASE4_WORKLOAD", str(fixture))
    monkeypatch.setenv("SR_PHASE4_TARGET_DIAGNOSTICS", str(output))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    proposal_tokens = tuple(getattr(pending, "proposal_token_ids", ()))
    query_tokens = [COMMITTED_PREFIX[-1], *proposal_tokens]
    query_length = len(query_tokens)
    stable_state = SimpleNamespace(
        pending_proposal=pending,
        committed_token_ids=COMMITTED_PREFIX,
    )
    input_batch = SimpleNamespace(
        req_ids=["internal-0"],
        num_tokens_no_spec=[len(COMMITTED_PREFIX)],
        token_ids_cpu=_FakeTensor(
            [[*COMMITTED_PREFIX, *proposal_tokens, *([0] * 4)]]
        ),
        num_computed_tokens_cpu=[len(COMMITTED_PREFIX) - 1],
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        drafter=SimpleNamespace(requests={REQUEST_ID: stable_state}),
        attn_groups=(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(disable_custom_all_reduce=True)
        ),
    )
    logits = _FakeTensor(
        [[float(token_id) for token_id in range(16)] for _ in query_tokens]
    )
    metadata = (
        SimpleNamespace(
            target_logits_indices=_FakeTensor(list(range(len(proposal_tokens)))),
            draft_token_ids=_FakeTensor(list(proposal_tokens)),
        )
        if proposal_tokens
        else None
    )
    capture_target_forward(
        runner,
        scheduler_output=SimpleNamespace(
            scheduled_spec_decode_tokens={"internal-0": list(proposal_tokens)}
        ),
        logits=logits,
        spec_decode_metadata=metadata,
        logits_indices=_FakeTensor(list(range(query_length))),
        positions=_FakeTensor(
            list(
                range(
                    len(COMMITTED_PREFIX) - 1,
                    len(COMMITTED_PREFIX) - 1 + query_length,
                )
            )
        ),
        num_scheduled_tokens=[query_length],
        common_attention_metadata=SimpleNamespace(
            query_start_loc_cpu=_FakeTensor([0, query_length]),
            seq_lens=_FakeTensor([len(COMMITTED_PREFIX) - 1 + query_length]),
            causal=True,
        ),
        target_forward_start_ns=100,
        target_forward_end_ns=200,
    )
    rows = CheckpointJsonl(output).read()
    assert len(rows) == 1
    return rows[0]


def test_diagnostic_proposal_identity_is_protocol_aware():
    serial = _serial_proposal()
    dual = _dual_proposal()
    assert diagnostic_proposal_id(None) is None
    assert diagnostic_proposal_id(serial) is None
    assert diagnostic_proposal_id(dual) == dual.proposal_id
    assert "proposal_id" not in serial.to_dict()


@pytest.mark.parametrize("pending", (None, _serial_proposal()))
def test_capture_target_forward_accepts_none_and_serial_proposal(
    tmp_path, monkeypatch, pending
):
    row = _capture(tmp_path, monkeypatch, pending)
    assert row["proposal_id"] is None
    assert row["round_id"] is (None if pending is None else 0)
    assert validate_target_diagnostic(row) == []


def test_serial_first_verification_remains_decode_ready_compatible(
    tmp_path, monkeypatch
):
    row = _capture(tmp_path, monkeypatch, _serial_proposal())
    request = DecodeReadyRequest(
        request_id=REQUEST_ID,
        internal_target_request_id="internal-0",
        prompt_token_count=3,
        prompt_token_ids_sha256=token_prefix_hash((1, 2, 3)),
        bootstrap_token_id=7,
        committed_output_token_count=1,
        logical_committed_prefix_count=4,
        logical_committed_prefix_sha256=token_prefix_hash(COMMITTED_PREFIX),
        logical_committed_prefix_token_ids=COMMITTED_PREFIX,
        target_materialized_kv_token_count=3,
        target_pending_input_token_id=7,
        target_pending_input_position=3,
        target_num_computed_tokens=3,
        target_num_computed_tokens_relation="prompt-only",
        draft_materialized_kv_token_count=4,
        prefix_version=1,
        next_round_id=0,
        target_decode_ready=True,
        draft_decode_ready=True,
        initial_proposal_generated=False,
        bootstrap_ready_ns=1,
        draft_initialization_complete_ns=2,
    )
    contract = build_first_target_forward_contract(
        request,
        consumer="serial",
        proposal_token_ids=row["proposal_token_ids"],
        target_forward_start_ns=row["target_forward_start_ns"],
        target_forward_end_ns=row["target_forward_end_ns"],
        output_logits_positions=row["position_ids"],
        accepted_draft_tokens=1,
        post_forward_committed_token_ids=(10, 9),
        post_forward_target_kv_token_count=5,
        post_forward_prefix_version=2,
    )
    assert row["target_input_token_ids"] == contract["verification_input_token_ids"]
    assert row["position_ids"] == contract["input_positions"]
    assert validate_first_target_forward_contract(contract, request) == []


def test_dual_capture_and_validator_keep_exact_canonical_proposal_id(
    tmp_path, monkeypatch
):
    proposal = _dual_proposal()
    diagnostic = _capture(tmp_path, monkeypatch, proposal)
    assert diagnostic["proposal_id"] == proposal.proposal_id
    assert validate_verification_contracts(
        [
            {
                "request_id": REQUEST_ID,
                "round_id": 0,
                "proposal_id": proposal.proposal_id,
                "proposal_token_ids": list(PROPOSAL_TOKENS),
            }
        ],
        [
            {
                "proposal_id": proposal.proposal_id,
                "proposal_token_ids": list(PROPOSAL_TOKENS),
            }
        ],
        [diagnostic],
    ) == []
