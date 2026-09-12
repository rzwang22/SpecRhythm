"""Bounded, opt-in host causality on the actual owner/protocol adapters."""

import itertools
from threading import Event

import pytest
from test_serial_eager_launch_diagnosis import resident_owner as _resident_owner
from test_serial_eager_owner import sync_row, verify_row

from specrhythm.continuation.trace import TRACE, CausalTrace
from specrhythm.serving.eager_latency import causal_cycle, exclusive_main, trace_rows

resident_owner = _resident_owner


@pytest.fixture
def tracing(monkeypatch):
    monkeypatch.setattr(TRACE, 'enabled', True)
    monkeypatch.setattr(TRACE, 'rows', [])
    monkeypatch.setattr(TRACE, 'attempts', {})
    monkeypatch.setattr(TRACE, 'sequence', itertools.count())
    yield TRACE


def test_trace_cap_nested_context_and_exception_do_not_leak():
    trace = CausalTrace(True, 3)
    with trace.span('outer', request_id='r'):
        with pytest.raises(ValueError), trace.span('inner', round_id=4):
            raise ValueError('original failure')
        trace.event('after')
    trace.event('omitted')
    result = trace.report()
    assert len(result['rows']) == 3 and result['dropped_rows'] == 1
    assert result['status'] == 'TRUNCATED'
    assert trace.rows[0]['round_id'] == 4 and 'round_id' not in trace.rows[1]
    assert not trace.local.fields
    off = CausalTrace()
    with off.span('unused'):
        off.event('unused')
    assert off.report()['status'] == 'DISABLED' and not off.rows


def test_main_partition_does_not_add_nested_rpc_other_threads_or_gpu():
    base = dict(pid=1, thread_id=7)
    host = [dict(base, category='target_step', start_ns=10, end_ns=70),
            dict(base, category='ipc', start_ns=20, end_ns=60),
            dict(base, category='output_commit', start_ns=70, end_ns=80),
            dict(pid=2, thread_id=8, category='ipc', start_ns=0, end_ns=100)]
    causal = [dict(base, category='coordinator_checkpoint', start_ns=80, end_ns=95)]
    result = exclusive_main(host, causal, 0, 100)
    assert result['exclusive_ms'] == {'unaccounted': 15/1e6, 'target_step': 60/1e6,
                                     'output_commit': 10/1e6, 'coordinator_checkpoint': 15/1e6}
    assert exclusive_main([], [], 0, 100)['status'] == 'MISSING'
    assert trace_rows({}, 0, 100)['status'] == 'MISSING'


def forward(a, b, *, purpose='target', rank=0):
    return dict(start_lower_ns=a, start_upper_ns=a+1, end_lower_ns=b-1, end_upper_ns=b,
                host_start_ns=a, host_launch_end_ns=b, gpu_event_ms=(b-a)/1e6, B=1,
                internal_request_ids=['sr-draft:r'], purpose=purpose, global_rank=rank)


def test_gpu_tp_union_and_progress_use_calibrated_bounds_not_host_overlap():
    devices = [dict(identity={'global_rank': r}, forwards=[forward(20, 50)]) for r in (0, 1)]
    # Host enqueue and dispatch may overlap Target; eager device starts after it.
    draft = {'forwards': [forward(60, 80, purpose='eager')]}
    result = causal_cycle(dict(start_ns=0, end_ns=100, request_ids=['r']), devices, draft, [])
    assert result['native_overlap'] == {'lower_ms': 0, 'upper_ms': 0}
    assert result['classification'] == 'LATE_AFTER_TARGET'
    assert result['denominators']['target_TP_union'] == {'lower_ms': 28/1e6,
                                                       'upper_ms': 30/1e6}
    assert result['steps_completed_by_target_end']['r']['possibly_completed'] == 0
    assert result['owner_queue']['status'] == 'MISSING'
    draft['forwards'] = [forward(30, 40, purpose='eager')]
    result = causal_cycle(dict(start_ns=0, end_ns=100, request_ids=['r']), devices, draft, [])
    assert result['native_overlap']['lower_ms'] == 8/1e6
    assert result['steps_completed_by_target_end']['r']['definitely_completed'] == 1
    # One missing rank must be explicitly unavailable, never reconstructed.
    assert causal_cycle(dict(start_ns=0, end_ns=100, request_ids=['r']), [], draft, [])[
        'timeline_status'] == 'MISSING_OR_NO_EAGER'


def test_real_owner_trace_orders_enqueue_dequeue_gpu_and_feedback(resident_owner, tracing,
                                                                 monkeypatch):
    case = resident_owner
    entered, release = Event(), Event()
    original = case.backend.step_gpu_continuations
    def controlled(works):
        entered.set()
        assert release.wait(5)  # deadlock guard, not a latency assertion
        return original(works)
    monkeypatch.setattr(case.backend, 'step_gpu_continuations', controlled)
    case.owner.call('eager_enqueue', {'requests': [verify_row(p) for p in case.proposals],
                                        '_causal_batch_id': 'target-batch-1'})
    assert entered.wait(5)
    rows = tracing.rows
    queued = next(r for r in rows if r['category'] == 'owner_queue_submit')
    dequeued = next(r for r in rows if r['category'] == 'owner_dequeue')
    assert queued['queue_submit_ns'] <= dequeued['start_ns']
    assert dequeued['batch_id'] == 'target-batch-1'
    assert dequeued['requests'][0]['proposal_id'] == case.proposals[0]['runtime_provenance'][
        'rolling_proposal_id']
    assert not any(r['category'] == 'protocol_parent_result' for r in rows)
    release.set()
    feedback = [sync_row(p, case.prefixes[p['request_id']], reject=True)[0]
                for p in case.proposals]
    case.owner.call('synchronize_and_batch_propose',
                    {'synchronizations': feedback, 'proposals': []})
    assert any(r['category'] == 'protocol_parent_result' for r in rows)
    settled = [r for r in rows if r['category'] == 'protocol_parent_settled']
    assert all(b['round_id'] == 0 and b['continuation_id']
               for r in settled for b in r['bindings'])
    assert any(r['category'] == 'physical_begin_gpu_continuations' for r in rows)
    assert any(r['category'] == 'eligibility_snapshot' for r in rows)
    assert case.owner.machine.counters['parent_rejections'] == 16
