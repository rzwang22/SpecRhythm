"""Validate/fence/publish batching on real physical adapters and resident owner."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from test_rolling_eager_gpu_backend import (
    FencedWorker,
    begin,
    complete,
    normal,
    plan_and_receipt,
    setup,
)
from test_serial_eager_launch_diagnosis import resident_owner as _resident_owner
from test_serial_eager_owner import proposal_row, sync_row, verify_row

from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving.eager_machine import EagerSerialMachine

resident_owner = _resident_owner


@pytest.mark.parametrize('terminal', [False, True])
def test_mixed_future_and_ordinary_rows_share_one_fenced_ragged_repair(terminal):
    backend, worker, core, prefixes = setup(SimpleNamespace(max_model_len=4096), count=3)
    proposals = {rid: normal(backend, core, rid) for rid in prefixes}
    works = {rid: begin(backend, core, proposals[rid], rid) for rid in ('req-0', 'req-1')}
    completions = {rid: complete(backend, work) for rid, work in works.items()}
    plans = {
        'req-0': plan_and_receipt(proposals['req-0'])[0],
        'req-1': plan_and_receipt(proposals['req-1'], tail=(901,))[0],
        'req-2': plan_and_receipt(proposals['req-2'], terminal=terminal)[0],
    }
    settlements = [(plans[rid], works.get(rid),
                    completions[rid].generated_tokens[1:] if rid == 'req-0' else None)
                   for rid in prefixes]
    before = len(worker.calls)
    materialize = worker.materialize
    def observed(rows, purpose):
        value = materialize(rows, purpose)
        assert all(backend.states[rid].prefix == prefix for rid, prefix in prefixes.items())
        assert all(backend.states[rid].next_round == 0 for rid in prefixes)
        assert not backend._gpu_rebases  # no result published before the batch fence
        return value
    worker.materialize = observed
    result = backend.rebase_gpu_parents(settlements)
    assert [(p, len(rows)) for p, rows in worker.calls[before:]] == [('commit', 2)]
    assert [len(r.suffix) for r in worker.calls[-1][1]] == [1, 2]
    assert result['req-0']['materialized_query_tokens'] == 0
    assert result['req-1']['materialized_query_tokens'] == 1
    assert result['req-2']['materialized_query_tokens'] == 2
    assert backend.metrics.syncs['eager_parent_settlement'] == 1
    assert backend.metrics.syncs['eager_rebase_complete'] == 1
    for rid, plan in plans.items():
        if terminal and rid == 'req-2':
            assert rid in backend.retired and f'sr-draft:{rid}' not in worker.memory
        else:
            state = backend.states[rid]
            assert state.prefix == plan.final_prefix and state.next_round == 1
            assert tuple(worker.memory[state.internal_id][:len(state.prefix)]) == state.prefix
    assert backend.rebase_gpu_parents(settlements) == result
    assert len(worker.calls) == before+1
    for work in works.values():
        with pytest.raises(ValueError, match='retired'):
            backend.step_gpu_continuation(work)
    backend.shutdown()
    assert not worker.memory and not backend._gpu_continuations


@pytest.mark.parametrize('invalid', ['version', 'bridge', 'omitted_work'])
def test_last_invalid_parent_prevents_entire_batch_mutation(invalid):
    backend, worker, core, _ = setup(SimpleNamespace(max_model_len=4096), count=2)
    proposals = [normal(backend, core, f'req-{i}') for i in range(2)]
    works = [begin(backend, core, p, p.request_id) for p in proposals]
    completions = [complete(backend, w) for w in works]
    plans = [plan_and_receipt(p)[0] for p in proposals]
    rows = [(p, w, c.generated_tokens[1:]) for p, w, c in zip(plans, works, completions)]
    if invalid == 'version':
        rows[-1] = (replace(plans[-1], round_id=9), works[-1], rows[-1][2])
    elif invalid == 'bridge':
        rows[-1] = (plans[-1], works[-1], (999,))
    else:
        rows[-1] = (plans[-1], None, None)
    before = (list(worker.calls), dict(backend.metrics.counters))
    with pytest.raises(ValueError):
        backend.rebase_gpu_parents(rows)
    assert (worker.calls, dict(backend.metrics.counters)) == before
    assert not backend._gpu_rebases and len(backend._gpu_continuations) == 2
    assert all(s.next_round == 0 for s in backend.states.values())
    backend.shutdown()
    assert not worker.memory


def test_batched_repair_failure_is_fenced_and_cannot_publish_or_restart():
    backend, worker, core, _ = setup(SimpleNamespace(max_model_len=4096), count=2)
    proposals = [normal(backend, core, f'req-{i}') for i in range(2)]
    works = [begin(backend, core, p, p.request_id) for p in proposals]
    for work in works:
        complete(backend, work)
    plans = [plan_and_receipt(p, accepted=1, tail=(901,))[0] for p in proposals]
    materialize = worker.materialize
    def fails_after_write(rows, purpose):
        materialize(rows, purpose)
        raise RuntimeError('device batch failed after submission')
    worker.materialize = fails_after_write
    with pytest.raises(RuntimeError, match='device batch failed'):
        backend.rebase_gpu_parents([(p, w, None) for p, w in zip(plans, works)])
    assert backend.failed and not worker.inflight and not backend._gpu_writing
    assert not backend._gpu_rebases and all(s.next_round == 0 for s in backend.states.values())
    with pytest.raises(RuntimeError, match='failed'):
        backend.step_gpu_continuations(works)
    backend.shutdown()
    assert not worker.memory


def test_eager_disabled_uses_the_original_batched_commit_and_generation_path():
    workers = [FencedWorker(), FencedWorker()]
    config = SimpleNamespace(max_model_len=4096)
    baseline = BatchedDraftStateMachine(VllmBatchedDraftBackend(config, worker=workers[0]))
    eager = EagerSerialMachine(RollingVllmDraftBackend(config, worker=workers[1]),
                               request_ids=('a', 'b'))
    eager.set_eager(False, 1)
    outputs = []
    for machine in (baseline, eager):
        for rid in ('a', 'b'):
            machine.initialize(rid, (10, 20), token_prefix_hash((10, 20)))
        proposals = machine.batch_propose([proposal_row(rid) for rid in ('a', 'b')])['proposals']
        if machine is eager:
            machine.verify_start([verify_row(p) for p in proposals])
        feedback, next_rows = [], []
        for p in proposals:
            row, final = sync_row(p, (10, 20), reject=True)
            feedback.append(row)
            next_rows.append(proposal_row(p['request_id'], final, round_id=1, remaining=97))
        if machine is eager:
            machine.prepare_synchronizations(feedback)
            machine.finish_synchronizations(feedback)
            result = machine.batch_propose(next_rows)
        else:
            result = machine.synchronize_and_batch_propose(feedback, next_rows)
        outputs.append([p['proposal_token_ids'] for p in result['proposals']])
    assert outputs[0] == outputs[1]
    assert workers[0].calls == workers[1].calls
    assert workers[0].metrics.syncs == workers[1].metrics.syncs
    assert not eager.works and eager.counters['admissions'] == 0
    baseline.backend.shutdown()
    eager.backend.shutdown()


def test_resident_owner_settles_b16_with_two_pool_audits(resident_owner, monkeypatch):
    case = resident_owner
    completed = Event()
    step = case.backend.step_gpu_continuations
    def observed(works):
        result = step(works)
        if all(v is not None for v in result.values()):
            completed.set()
        return result
    monkeypatch.setattr(case.backend, 'step_gpu_continuations', observed)
    case.owner.call('eager_enqueue', {'requests': [verify_row(p) for p in case.proposals]})
    assert completed.wait(2)
    case.owner.call('status', {})
    feedback = []
    for i, p in enumerate(case.proposals):
        row, final = sync_row(p, case.prefixes[p['request_id']], reject=i>=5)
        if i >= 13:
            delta = tuple(p['proposal_token_ids']) + (901,)
            row.update(committed_delta=list(delta), committed_prefix_hash=token_prefix_hash(
                case.prefixes[p['request_id']] + delta))
        feedback.append(row)
    before_audits, before_visits = case.backend.audit_calls, case.backend.prefix_visits
    before_calls = len(case.worker.calls)
    case.owner.call('synchronize_and_batch_propose', {
        'synchronizations': feedback, 'proposals': []})
    # Two settlement audits plus the unchanged empty normal-proposal audit pair.
    assert case.backend.audit_calls-before_audits == 4
    assert case.backend.prefix_visits-before_visits == 4*360
    assert [(p, len(rows)) for p, rows in case.worker.calls[before_calls:]] == [('commit', 11)]
    assert case.owner.machine.counters['promotions'] == 5
    assert case.owner.machine.counters['parent_rejections'] == 8
    assert case.owner.machine.counters['bridge_mismatches'] == 3
    assert case.owner.machine.counters['committed_tokens'] == 64
