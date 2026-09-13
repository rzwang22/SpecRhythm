"""CPU substitutes only at GPU/IPC boundaries, actual production adapter/owner chain."""

from __future__ import annotations

import time
from types import SimpleNamespace as NS

from test_prepost_protocol import Backend
from test_prepost_target import make_runner
from test_prepost_target import runner_class as _runner_class
from test_serial_eager_owner import OwnerWorker, proposal_row, settle_row
from test_serving_fixed_startup import (
    hardware as _hardware,
)
from test_serving_fixed_startup import (
    observed_startup as _observed_startup,
)
from test_serving_fixed_startup import (
    phase4_config as _config,
)
from test_serving_fixed_startup import (
    startup as _startup,
)

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_remote import _TargetRequest
from specrhythm.serving import s2_runtime
from specrhythm.serving.eager_owner import EagerOwner
from specrhythm.serving.prepost_machine import PrePostMachine
from specrhythm.serving.prepost_target import install
from specrhythm.serving.s2_pool import publish

hardware, startup, phase4_config = _hardware, _startup, _config
observed_startup, runner_class = _observed_startup, _runner_class


def test_stock_bookkeeping_adapter_owner_mixed_batch_roundtrip(
    observed_startup, runner_class, monkeypatch
):
    from test_prepost_target import Array, np

    h = observed_startup
    monkeypatch.setitem(s2_runtime.CLASSES, "serial", s2_runtime.CLASSES["serial-prepost3"])
    h.run("serial")  # Real config/proposer construction with fixture hardware.
    proposer = h.workers[0].model_runner.drafter
    definitions = list(proposer.definitions.values())[:2]
    ids = [d.request_id for d in definitions]
    owner = EagerOwner(
        lambda: PrePostMachine(
            Backend(NS(max_model_len=4096), worker=OwnerWorker()), request_ids=ids
        ),
        timeout_seconds=3,
    )

    def rpc(operation, payload):
        value = owner.call(operation, payload)
        return {
            **value,
            "service_send_ns": time.monotonic_ns(),
            "transport_end_ns": time.monotonic_ns(),
        }

    proposer.client = NS(call=rpc)
    runner = make_runner(runner_class)
    runner.drafter = proposer
    runner.max_model_len = 4096
    runner.input_batch.token_ids_cpu = np.zeros((2, 4096), int)
    runner.input_batch.is_token_ids = np.ones((2, 4096), bool)
    runner.input_batch.req_ids = ["a", "b"]
    packet = {"requests": {}, "initial_proposals": {}}
    try:
        for i, (internal, d) in enumerate(zip(("a", "b"), definitions)):
            prefix = d.prompt_token_ids + (100,)
            proposer.identity.bind(internal, prefix)
            proposer.requests[d.request_id] = _TargetRequest(
                d.request_id,
                d.prompt_token_ids,
                d.maximum_new_tokens,
                prefix,
                (100,),
                bootstrap_target_tokens=1,
            )
            runner.requests[internal] = NS(
                output_token_ids=[100], sampling_params=NS(max_tokens=d.maximum_new_tokens)
            )
            runner.input_batch.num_tokens_no_spec[i] = len(prefix)
            runner.input_batch.token_ids_cpu[i, : len(prefix)] = list(prefix)
            rpc(
                "initialize",
                dict(
                    request_id=d.request_id,
                    committed_token_ids=list(prefix),
                    committed_prefix_hash=token_prefix_hash(prefix),
                ),
            )
            initial = rpc(
                "batch_propose",
                {"requests": [proposal_row(d.request_id, prefix, 0, d.maximum_new_tokens - 1)]},
            )
            packet["requests"][d.request_id] = {"state": "ACTIVE"}
            packet["initial_proposals"][d.request_id] = dict(
                proposal=initial["proposals"][0],
                service_send_ns=initial["service_send_ns"],
                transport_end_ns=initial["transport_end_ns"],
            )
        publish(h.directory / "s2-control.json", packet)
        install(runner)
        expected_lengths = [[1, 1], [4, 4], [4, 1], [4, 1], [4, 4], [4, 4]]
        for cycle in range(6):
            proposals = [owner.machine.requests[rid].pending_proposal for rid in ids]
            scheduled = {
                internal: list(p.proposal_token_ids) for internal, p in zip(("a", "b"), proposals)
            }
            assert [len(v) for v in scheduled.values()] == expected_lengths[cycle]
            proposer.on_target_verify_start(
                request_ids=["a", "b"], scheduled_spec_token_ids=scheduled
            )
            raw = []
            for i, p in enumerate(proposals):
                values = (
                    [998] if i == 1 and cycle in (1, 2) else list(p.proposal_token_ids) + [777]
                )
                raw.append(values + [-1] * (5 - len(values)))
            output = runner._bookkeeping_sync(
                NS(
                    scheduled_spec_decode_tokens=scheduled,
                    num_scheduled_tokens={k: len(v) + 1 for k, v in scheduled.items()},
                ),
                NS(sampled_token_ids=Array(raw), logprobs_tensors=None),
                None,
                [0] * 10,
                10,
            )
            proposer.on_target_verify_end(
                request_ids=["a", "b"],
                sampled_token_ids=output[2],
                scheduled_spec_token_ids=scheduled,
            )
            result = proposer._rank_zero_propose(
                ["a", "b"], runner.input_batch.num_tokens_no_spec, runner.input_batch.token_ids_cpu
            )
            assert len(result["draft_token_ids"]) == 2
            for internal, rid in zip(("a", "b"), ids):
                assert proposer.requests[rid].generated_token_ids == tuple(
                    runner.requests[internal].output_token_ids
                )
                assert (
                    owner.machine.requests[rid].committed_token_ids
                    == proposer.requests[rid].committed_token_ids
                )
        assert len(proposer.round_records) == 12
        assert all(r["target_bonus_tokens"] == 0 for r in proposer.round_records)
        assert all(r["prepost_protocol"] for r in proposer.round_records)
        assert owner.machine.backend.metrics.forwards["prepost_post"] == 6
        assert owner.machine.backend.metrics.forwards["proposal"] == 0
    finally:
        for rid, state in owner.machine.requests.items():
            owner.call(
                "diagnostic_settle",
                settle_row(
                    rid, state.committed_token_ids, state.next_round_id, terminal=state.finished
                ),
            )
        owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
