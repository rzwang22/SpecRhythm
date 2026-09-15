"""CPU substitutes only at GPU/IPC boundaries, actual production adapter/owner chain."""

from __future__ import annotations

import time
from types import SimpleNamespace as NS

from test_ping_prepost import Backend
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
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine
from specrhythm.serving.ping_prepost_owner import PingPrePostOwner
from specrhythm.serving.ping_prepost_proposer import FeedbackClient
from specrhythm.serving.prepost_target import install
from specrhythm.serving.s2_pool import publish

hardware, startup, phase4_config = _hardware, _startup, _config
observed_startup, runner_class = _observed_startup, _runner_class


def test_stock_bookkeeping_ping_adapter_controller_owner_mixed_batch_roundtrip(
    observed_startup, runner_class, monkeypatch
):
    from test_prepost_target import Array, np

    h = observed_startup
    monkeypatch.setitem(
        s2_runtime.CLASSES, "serial", s2_runtime.CLASSES["pingpong-eager-prepost3"]
    )
    h.run("serial")  # Real config/proposer construction with fixture hardware.
    proposer = h.workers[0].model_runner.drafter
    definitions = list(proposer.definitions.values())[:2]
    ids = [d.request_id for d in definitions]
    owner = PingPrePostOwner(
        lambda: PingPrePostMachine(
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

    proposer.client = FeedbackClient(NS(call=rpc))
    controller = PingPrePostController()
    import threading

    ready = {rid: threading.Event() for rid in ids}
    publish_proposal = owner.machine._publish_proposal

    def observed_publish(rid, *args, **kwargs):
        result = publish_proposal(rid, *args, **kwargs)
        ready[rid].set()
        return result

    owner.machine._publish_proposal = observed_publish
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
            rpc(
                "pp_register",
                {
                    "requests": [
                        dict(
                            proposal_row(d.request_id, prefix, 0, d.maximum_new_tokens - 1),
                            home_cohort="A",
                        )
                    ]
                },
            )
            packet["requests"][d.request_id] = {"state": "ACTIVE"}
        publish(h.directory / "s2-control.json", packet)
        install(runner)
        expected_lengths = [[1, 1], [4, 4], [4, 1], [4, 1], [4, 4], [4, 4]]
        for cycle in range(6):
            assert all(e.wait(3) for e in ready.values())
            for event in ready.values():
                event.clear()
            # Real controller, owner atomic claim, then actual Target adapter.
            # Keep opportunity A for mixed recovery; a separate protocol regression
            # exercises consecutive A/B/A promoted claims.
            controller.admitted = cycle * 2
            packet["pp_admission"] = controller.select(NS(rows=packet["requests"]), NS(call=rpc))
            publish(h.directory / "s2-control.json", packet)
            assert len(packet["pp_admission"]["claims"]) == 2
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
            assert result["draft_token_ids"] == [[], []]
            assert all(e.wait(3) for e in ready.values())
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
        assert all(r["ping_prepost_protocol"] for r in proposer.round_records)
        assert all(r["logical_draft_kv_length"] is None for r in proposer.round_records)
        assert (
            sum(
                1
                for f in owner.machine.backend.prepost_forwards.rows()
                if any(b.get("work_kind") == "post" for b in f["bindings"])
            )
            >= 6
        )
        assert owner.machine.backend.metrics.forwards["proposal"] == 0
    finally:
        owner.call("pp_stop", {})
        for rid, state in owner.machine.requests.items():
            owner.call(
                "diagnostic_settle",
                settle_row(
                    rid, state.committed_token_ids, state.next_round_id, terminal=state.finished
                ),
            )
        owner.call("shutdown", {"deadline_ns": time.monotonic_ns() + 3_000_000_000})
