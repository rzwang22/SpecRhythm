"""Actual Target scheduler/verify adapter/queue/backend; GPU bodies are event-controlled."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from test_k3 import Backend
from test_k3_pipeline import close
from test_k3_prompt_proof import (
    fixed_schedulers as _fixed,
)
from test_k3_prompt_proof import (
    s2_schedulers as _s2,
)
from test_k3_prompt_proof import (
    target_pool as _pool,
)
from test_ping_prepost import run_work
from test_ping_prepost_integration import (
    hardware as _hardware,
)
from test_ping_prepost_integration import (
    observed_startup as _observed,
)
from test_ping_prepost_integration import (
    phase4_config as _config,
)
from test_ping_prepost_integration import (
    startup as _startup,
)
from test_prepost_protocol import feedback
from test_serial_eager_owner import OwnerWorker, proposal_row

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_remote import _TargetRequest
from specrhythm.serving import s2_pool, s2_runtime
from specrhythm.serving.k3_machine import K3Machine
from specrhythm.serving.k3_owner import K3Owner
from specrhythm.serving.ping_prepost_controller import PingPrePostController
from specrhythm.serving.ping_prepost_proposer import FeedbackClient
from specrhythm.serving.s2_pool import publish

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool
hardware, observed_startup, phase4_config, startup = _hardware, _observed, _config, _startup


@pytest.mark.parametrize("diagnostics", ["full", "lean"])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("target_first", [False, True])
def test_actual_claim_schedule_verify_enters_before_other_recovery_finishes(
    observed_startup, target_pool, monkeypatch, eager, target_first, diagnostics
):
    h = observed_startup
    monkeypatch.setenv("SR_FIXED_TARGET_DIAGNOSTICS", diagnostics)
    monkeypatch.setenv("SR_FIXED_POINT", str(h.directory / "cpu-point.json"))
    monkeypatch.setitem(s2_runtime.CLASSES, "serial", s2_runtime.CLASSES["pingpong-eager-k3"])
    h.run("serial")  # Real proposer factory; only fixture GPU/transport replaced.
    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_S2_MODE", "pingpong-eager-k3" if eager else "pingpong-k3")
    packet["max_requests_per_target_forward"] = 8
    for i, row in enumerate(packet["requests"].values()):
        row["cohort"] = "A" if i < 8 else "B"
    publish(path, packet)
    s.freeze_pool()
    proposer = h.workers[0].model_runner.drafter
    proposer.identity = s._identity()
    proposer.internal_to_stable = proposer.identity.internal_to_stable
    proposer.requests = {}
    entered, resume, a_ready, target_entered, target_finish = [threading.Event() for _ in range(5)]
    second, resume_second, queued = [threading.Event() for _ in range(3)]

    class Worker(OwnerWorker):
        armed = False

        def materialize(self, rows, purpose):
            if (
                self.armed
                and purpose in ("prepost_extension", "prepost_mixed")
                and not entered.is_set()
            ):
                entered.set()
                assert resume.wait(3)
            elif (
                self.armed
                and purpose in ("prepost_extension", "prepost_mixed")
                and not second.is_set()
            ):
                second.set()
                assert resume_second.wait(3)
            return super().materialize(rows, purpose)

    worker = Worker()
    ids = list(s.requests)[:16]

    def factory():
        m = K3Machine(Backend(NS(max_model_len=4096), worker=worker), request_ids=ids, eager=eager)
        for rid in ids:
            r = s.requests[rid]
            r.sampling_params = NS(max_tokens=100)
            prefix = tuple(r.all_token_ids)
            m.initialize(rid, prefix, token_prefix_hash(prefix))
            proposer.requests[rid] = _TargetRequest(
                rid,
                tuple(r.prompt_token_ids),
                100,
                prefix,
                prefix[len(r.prompt_token_ids) :],
                bootstrap_target_tokens=1,
            )
        m.register(
            [
                dict(
                    proposal_row(rid, tuple(s.requests[rid].all_token_ids)),
                    home_cohort=packet["requests"][rid]["cohort"],
                )
                for rid in ids
            ]
        )
        run_work(m)
        original = m._ping

        def record(event, **values):
            original(event, **values)
            if event == "ready" and values["request_id"] == "0" and values["prefix_version"] == 1:
                a_ready.set()

        m._ping = record
        return m

    owner = K3Owner(factory, timeout_seconds=3)
    put = owner.commands.put

    def enqueue(command, **kwargs):
        put(command, **kwargs)
        if command[0] == "pp_admit" and command[1]["opportunity"] == 1:
            queued.set()

    owner.commands.put = enqueue
    proposer.client = FeedbackClient(NS(call=owner.call))
    controller = PingPrePostController()
    hashes = []
    original = s2_pool.token_prefix_hash
    prompts = {tuple(r.prompt_token_ids) for r in s.requests.values()}

    def hashed(tokens):
        if tuple(tokens) in prompts:
            hashes.append(tuple(tokens))
        return original(tokens)

    monkeypatch.setattr(s2_pool, "token_prefix_hash", hashed)

    def dispatch():
        packet["pp_admission"] = controller.select(
            NS(rows=packet["requests"]), NS(call=owner.call)
        )
        publish(path, packet)
        result = s.schedule()
        proposer.on_target_verify_start(
            request_ids=list(result.num_scheduled_tokens),
            scheduled_spec_token_ids=result.scheduled_spec_decode_tokens,
        )
        return result

    try:
        a = dispatch()
        assert set(a.num_scheduled_tokens) == set(ids[:8])
        # Sampling completed on A; actual adapter receives it, authoritative owner
        # gets the same rejection/acceptance inputs. Only the GPU sampler is absent.
        proposer.on_target_verify_end(
            request_ids=list(a.num_scheduled_tokens),
            sampled_token_ids=[],
            scheduled_spec_token_ids=a.scheduled_spec_decode_tokens,
        )
        rows = []
        for c in packet["pp_admission"]["claims"]:
            rid = c["request_id"]
            row, final = feedback(
                c["proposal"], owner.machine.requests[rid].committed_token_ids, reject=rid == "0"
            )
            rows.append(row)
            r = s.requests[rid]
            r.all_token_ids[:] = final
            r.num_output_tokens = len(final) - r.num_prompt_tokens
            r.num_computed_tokens = len(final) - 1
        worker.armed = True
        proposer.client.call("synchronize_and_batch_propose", {"synchronizations": rows})
        assert entered.wait(3)  # A's actual first extension is in flight.
        before_checks = s.s2_pool.checks

        def target_b():
            b = dispatch()
            assert set(b.num_scheduled_tokens) == set(ids[8:])
            target_entered.set()  # Reached execution after REAL scheduler + verify hook.
            assert target_finish.wait(3)
            proposer.on_target_verify_end(
                request_ids=list(b.num_scheduled_tokens),
                sampled_token_ids=[],
                scheduled_spec_token_ids=b.scheduled_spec_decode_tokens,
            )

        # Claim waits only for the in-flight fenced step; the next extension is
        # held until B has passed its actual scheduler and verify-start adapter.
        with ThreadPoolExecutor(1) as pool:
            f = pool.submit(target_b)
            assert queued.wait(3)
            resume.set()
            assert second.wait(3)
            assert target_entered.wait(3)
            assert not a_ready.is_set() and not resume_second.is_set()
            if target_first:
                target_finish.set()
                f.result(timeout=3)
            resume_second.set()
            assert a_ready.wait(3)
            target_finish.set()
            f.result(timeout=3)
        assert s.s2_pool.checks == before_checks + 2
        assert not hashes  # Old scheduler rehashes all resident360 twice per dispatch.
        assert len(owner.machine.ready["0"]["proposal"]["proposal_token_ids"]) == 3
    finally:
        resume.set()
        target_finish.set()
        resume_second.set()
        close(owner)
