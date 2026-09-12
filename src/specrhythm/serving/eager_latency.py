"""Window/step/device causality analysis; inclusive CPU lanes are not additive."""

from __future__ import annotations

from collections import Counter

from specrhythm.serving.eager_evidence_export import (
    bounds,
    device_summary,
    duration,
    host_summary,
    intersect,
    touching,
)
from specrhythm.serving.fixed_results import stats


def trace_rows(host, start, end):
    trace = host.get('causal_timeline', {})
    return {
        **{k: v for k, v in trace.items() if k != 'rows'},
        'status': trace.get('status', 'MISSING'),
        'rows': [r for r in trace.get('rows', []) if touching(r, start, end)],
    }


def exclusive_main(host, causal, start, end):
    """Disjoint accounting on the coordinator's engine.step thread only.

    A subprocess's CPU or GPU work is never subtracted from the coordinator's
    synchronous engine/RPC call. Those nested dependencies use separate lanes.
    """
    steps = [r for r in host if r['category'] == 'target_step']
    lanes = {(r.get('pid'), r.get('thread_id')) for r in steps}
    if len(lanes) != 1 or None in next(iter(lanes), (None,)):
        return {'status': 'MISSING', 'reason': 'coordinator step PID/thread not recorded'}
    lane = next(iter(lanes))
    priority = ['output_commit', 'coordinator_checkpoint', 'target_step', 'wait_draft_owner',
                'refill', 'initial_draft', 'control_json_write', 'control_json_read', 'ipc']
    rows = [r for r in host+causal if (r.get('pid'), r.get('thread_id')) == lane
            and r['category'] in priority and touching(r, start, end)]
    edges = sorted({start, end} | {max(start, r['start_ns']) for r in rows}
                   | {min(end, r['end_ns']) for r in rows})
    totals = Counter()
    for a, b in zip(edges, edges[1:]):
        active = {r['category'] for r in rows if r['start_ns'] <= a and r['end_ns'] >= b}
        label = next((p for p in priority if p in active), 'unaccounted')
        totals[label] += b-a
    return {'status': 'COMPLETE', 'pid': lane[0], 'thread_id': lane[1],
            'window_ms': (end-start)/1e6, 'exclusive_ms': {k: v/1e6 for k, v in totals.items()},
            'precedence': priority, 'semantics': 'disjoint coordinator host wall only; '
            'target_step includes child-process RPC/GPU dependencies; unaccounted is not idle'}


def native(device, start, end):
    result = device_summary(device, start, end, 100000)
    if result.get('status') == 'MISSING':
        return result
    # Preserve explicit work/parent/token-step joins from the model hook, as well
    # as the existing calibrated native endpoints and source-row fingerprint.
    for row in result['forwards']:
        original = device['forwards'][row['source_forward_index']]
        if 'causal_context' in original:
            row['causal_context'] = original['causal_context']
    return result


def endpoint_delta(first, second, endpoint):
    return {'lower_ms': (first['start_lower_ns']-second[endpoint+'_upper_ns'])/1e6,
            'upper_ms': (first['start_upper_ns']-second[endpoint+'_lower_ns'])/1e6}


def causal_cycle(step, target_devices, draft, traces):
    start, end = step['start_ns'], step['end_ns']
    targets = {str(d['identity']['global_rank']): native(d, start, end)
               for d in target_devices}
    target = [r for d in targets.values() for r in d.get('forwards', [])]
    eager = [r for r in draft['forwards'] if r.get('purpose') == 'eager'
             and start <= r['host_start_ns'] <= end]
    rows = [r for r in traces if touching(r, start, end)]
    result = {'target_ranks': targets, 'request_batch': len(step['request_ids']),
              'timeline_status': ('COMPLETE' if len(targets) == 2 and eager
                                  and all(d.get('forwards') for d in targets.values())
                                  else 'MISSING_OR_NO_EAGER'),
              'host_events': rows}
    if len(targets) != 2 or any(not d.get('forwards') for d in targets.values()) or not eager:
        return result
    first = min(eager, key=lambda r: r['start_lower_ns'])
    target_span = {f'{edge}_{side}_ns': fn(r[f'{edge}_{side}_ns'] for r in target)
                   for edge, fn in [('start', min), ('end', max)]
                   for side in ('lower', 'upper')}
    overlap = {side+'_ms': duration(intersect(bounds(target, start, end, inner),
                                             bounds(eager, start, end, inner)))
               for side, inner in [('lower', True), ('upper', False)]}
    target_union = {side+'_ms': duration(bounds(target, start, end, inner))
                    for side, inner in [('lower', True), ('upper', False)]}
    eager_union = {side+'_ms': duration(bounds(eager, start, end, inner))
                   for side, inner in [('lower', True), ('upper', False)]}
    progress = {}
    for f in eager:
        for internal in f.get('internal_request_ids', []):
            rid = internal.removeprefix('sr-draft:')
            counts = progress.setdefault(rid, {'definitely_completed': 0, 'possibly_completed': 0})
            counts['definitely_completed'] += int(f['end_upper_ns'] <= target_span['end_lower_ns'])
            counts['possibly_completed'] += int(f['end_lower_ns'] <= target_span['end_upper_ns'])
    dequeues = [r for r in rows if r['category'] == 'owner_dequeue'
                and r.get('operation') == 'eager_enqueue']
    queue = {'status': 'MISSING', 'reason': 'need one owner dequeue and submit marker'}
    if len(dequeues) == 1 and type(dequeues[0].get('queue_submit_ns')) is int:
        d = dequeues[0]
        queue = {'status': 'OBSERVED', 'queue_submit_ns': d['queue_submit_ns'],
                 'owner_dequeue_ns': d['start_ns'],
                 'submit_to_dequeue_ms': (d['start_ns']-d['queue_submit_ns'])/1e6,
                 'dequeue_to_first_gpu': {
                     'lower_ms': (first['start_lower_ns']-d['start_ns'])/1e6,
                     'upper_ms': (first['start_upper_ns']-d['start_ns'])/1e6},
                 'semantics': 'submit marker immediately before queue.put; includes put cost; '
                 'raw queue_put span brackets insertion, not an atomic dequeue clock'}
    copies = [r for r in rows if r['category'] == 'eligibility_snapshot']
    before = [r for r in copies if r['end_ns'] <= first['host_start_ns']]
    outcome = ('LATE_AFTER_TARGET' if first['start_lower_ns'] >= target_span['end_upper_ns']
               else 'RECORDED_OVERLAP' if overlap['lower_ms'] > 0 else 'BOUNDARY_UNCERTAIN')
    # Per-request outcome events are retained alongside progress. Batch overlap
    # alone must not be labelled as wasted work for every request in that batch.
    result.update({
        'classification': outcome, 'first_eager_relative_target_start': endpoint_delta(
            first, target_span, 'start'),
        'first_eager_relative_target_end': endpoint_delta(first, target_span, 'end'),
        'target_envelope_bounds_ns': target_span, 'native_overlap': overlap,
        'denominators': {'step_wall_ms': (end-start)/1e6, 'target_TP_union': target_union,
                         'eager_union': eager_union},
        'steps_completed_by_target_end': progress, 'owner_queue': queue,
        'eligibility_snapshot': {
            'status': 'OBSERVED' if copies else 'MISSING',
            'count': len(copies), 'before_first_forward_count': len(before),
            'union_ms': duration([(r['start_ns'], r['end_ns']) for r in copies]),
            'before_first_forward_union_ms': duration([
                (r['start_ns'], r['end_ns']) for r in before]),
            'copied_history_objects': sum(r.get('copied_history_objects', 0) for r in copies),
            'max_history_proposals': max((r['history_proposals'] for r in copies), default=None),
            'max_history_continuations': max((r['history_continuations'] for r in copies),
                                            default=None),
        },
    })
    return result


def latency_analysis(runtime, backend, steps, start, end):
    target_devices = [r['device'] for r in runtime['target_devices']]
    lane_hosts = {'coordinator': runtime.get('host', {}), 'draft': backend.get('fixed_host', {})}
    lane_hosts.update({
        'target-rank-'+str(r['device']['identity']['global_rank']): r.get('host', {})
                       for r in runtime['target_devices']})
    traces = {name: trace_rows(host, start, end) for name, host in lane_hosts.items()}
    all_trace = [r for h in traces.values() for r in h['rows']]
    wall = [(s['start_ns'], s['end_ns']) for s in steps]
    draft = backend['fixed_device']
    targets = [r for d in target_devices for r in d['forwards']]
    eager = [r for r in draft['forwards'] if r.get('purpose') == 'eager']
    normal = [r for r in traces['draft']['rows'] if r['category'] == 'normal_recovery_proposal']
    groups = {}
    for category in ('normal', 'recovery', 'mixed', 'empty'):
        selected = [r for r in normal if (
            next(iter(set(r['request_kinds'].values()))) if len(set(
                r['request_kinds'].values())) == 1 else 'mixed' if r['request_kinds']
            else 'empty') == category]
        spans = [(r['start_ns'], r['end_ns']) for r in selected]
        groups[category] = {'batch_count': len(selected), 'host_union_ms': duration(spans),
                            'status': 'OBSERVED' if selected else 'NOT_OBSERVED',
                            'gpu_union': {side+'_ms': duration(intersect(
                                bounds([f for f in draft['forwards']
                                        if f.get('purpose') == 'proposal'],
                                       start, end, inner), spans))
                                for side, inner in [('lower', True), ('upper', False)]}}
    return {
        'schema_version': 'specrhythm.eager-critical-path.v1',
        'three_clocks': {
            'window_wall_ms': (end-start)/1e6,
            'complete_step_wall_ms': stats([(b-a)/1e6 for a, b in wall]),
            'step_union_ms': duration(wall),
            'outside_complete_steps_ms': (end-start)/1e6-duration(wall),
            'target_gpu_TP_union': {side+'_ms': duration(bounds(targets, start, end, inner))
                                    for side, inner in [('lower', True), ('upper', False)]},
            'semantics': 'engine.step start through output commit is host step wall; '
            'continuous window includes step-external checks; GPU is native TP union'},
        'target_ranks': {str(d['identity']['global_rank']): native(d, start, end)
                         for d in target_devices},
        'draft_device': native(draft, start, end),
        'native_eager_overlap': {side+'_ms': duration(intersect(
            bounds(targets, start, end, inner), bounds(eager, start, end, inner)))
            for side, inner in [('lower', True), ('upper', False)]},
        'coordinator_exclusive': exclusive_main(runtime.get('host', {}).get('intervals', []),
                                               traces['coordinator']['rows'], start, end),
        'inclusive_cpu_lanes': {name: host_summary(host.get('intervals'), start, end, 100000)
                                for name, host in lane_hosts.items()},
        'trace_coverage': {name: {k: v for k, v in t.items() if k != 'rows'}
                           for name, t in traces.items()},
        'normal_recovery': groups,
        'cycles': [causal_cycle(s, target_devices, draft, all_trace) for s in steps],
        'event_lanes': traces,
        'batch_histogram': dict(Counter(str(len(s['request_ids'])) for s in steps)),
        'limitations': ['native model forwards only; not all GPU kernels',
                       'host CPU lanes, RPC wall and GPU costs overlap; do not add',
                       'missing or truncated traces cannot prove no wait/copy',
                       'mixed normal/recovery batch cost cannot be split by request',
                       'queue/GIL/lock cause requires explicit wait evidence; not inferred'],
    }
