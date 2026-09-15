"""Pinned stock bookkeeping + actual Target projection + real K3 backend full outputs."""

from types import SimpleNamespace as NS

import pytest
from test_k3 import machine
from test_ping_prepost import admit, run_work
from test_prepost_protocol import cleanup
from test_prepost_target import make_runner, np
from test_prepost_target import runner_class as _runner_class
from test_rolling_eager_gpu_backend import greedy_tokens
from test_serial_eager_owner import OwnerWorker, verify_row

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.k3 import MODES, geometry
from specrhythm.serving.prepost_target import install

runner_class = _runner_class


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("feedback_first", [False, True])
def test_complete_output_against_independent_target_oracle(
    mode, feedback_first, runner_class, monkeypatch
):
    class Draft(OwnerWorker):
        def greedy(self, logits):
            values = super().greedy(logits)
            return [t + int(t % 3 == 0) for t in values]  # deterministic imperfect Draft

    monkeypatch.setenv("SR_S2_MODE", mode)
    g = geometry(mode)
    m = machine(g["eager"], ids=tuple(str(i) for i in range(16)), budget=31,
                worker=Draft(), mode=mode)
    seen_rejection = False
    try:
        for cycle in range(80):
            active = [rid for rid, s in m.requests.items() if not s.finished]
            if not active:
                break
            claims = admit(m, "A" if g["serial_idle_gate"] or cycle % 2 == 0 else "B",
                           g["target_request_ceiling"], active)["claims"]
            ids = [c["request_id"] for c in claims]
            assert 0 < len(ids) <= g["target_request_ceiling"]
            m.verify_start([verify_row(c["proposal"]) for c in claims])
            if not feedback_first:
                run_work(m)
            runner = make_runner(runner_class)
            runner.max_model_len = 128
            runner.drafter.eos_token_ids = ()
            runner.input_batch.num_reqs = len(ids)
            runner.discard_request_mask.np = np.array([False] * len(ids))
            runner.input_batch.req_ids = ids
            runner.input_batch.req_id_to_index = {rid: i for i, rid in enumerate(ids)}
            runner.input_batch.token_ids_cpu = np.zeros((len(ids), 128), int)
            runner.input_batch.is_token_ids = np.ones((len(ids), 128), bool)
            runner.requests = {
                rid: NS(
                    output_token_ids=list(m.requests[rid].committed_token_ids[1:]),
                    sampling_params=NS(max_tokens=32),
                )
                for rid in ids
            }
            runner.input_batch.num_tokens_no_spec = np.array(
                [len(m.requests[rid].committed_token_ids) for rid in ids]
            )
            proposals = {c["request_id"]: c["proposal"]["proposal_token_ids"] for c in claims}
            raw = []
            for rid in ids:
                prefix = m.requests[rid].committed_token_ids
                delta = []
                for token in proposals[rid]:
                    correct = greedy_tokens(prefix + tuple(delta), 1)[0]
                    delta.append(correct)
                    if correct != token:
                        seen_rejection = True
                        break
                else:
                    delta.append(greedy_tokens(prefix + tuple(delta), 1)[0])  # unused stock bonus
                raw.append(delta + [-1] * (4 - len(delta)))
            install(runner)
            sampled = runner._bookkeeping_sync(
                NS(
                    scheduled_spec_decode_tokens=proposals,
                    num_scheduled_tokens={rid: len(proposals[rid]) + 1 for rid in ids},
                ),
                NS(sampled_token_ids=np.array(raw), logprobs_tensors=None),
                None,
                [0] * 8,
                8,
            )[2]
            feedback = []
            for rid, delta in zip(ids, sampled):
                final = m.requests[rid].committed_token_ids + tuple(delta)
                feedback.append(
                    dict(
                        request_id=rid,
                        round_id=m.requests[rid].next_round_id,
                        committed_delta=delta,
                        committed_prefix_hash=token_prefix_hash(final),
                        terminal=len(final) == 33,
                    )
                )
                assert runner.requests[rid].output_token_ids == list(final[1:])
            m.feedback(feedback)
            run_work(m)
        assert seen_rejection and all(s.finished for s in m.requests.values())
        expected = (10, 20) + greedy_tokens((10, 20), 31)
        assert all(s.committed_token_ids == expected for s in m.requests.values())
        assert m.counters["committed_tokens"] == 16 * 31
        assert (
            m.counters["committed_tokens"]
            == m.counters["accepted_tokens"] + m.counters["correction_tokens"]
        )
    finally:
        cleanup(m)


def test_target_k3_rejects_nontail_short_candidates(runner_class, monkeypatch):
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    runner = make_runner(runner_class)
    install(runner)
    with pytest.raises(ValueError, match="incomplete K3"):
        runner._bookkeeping_sync(
            NS(
                scheduled_spec_decode_tokens={"a": [2], "b": [3]},
                num_scheduled_tokens={"a": 2, "b": 2},
            ),
            NS(sampled_token_ids=np.array([[2, 4], [3, 4]]), logprobs_tensors=None),
            None,
            [0] * 4,
            4,
        )
