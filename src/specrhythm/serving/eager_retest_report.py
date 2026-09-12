"""Offline B16 repair attribution from ended, immutable production points.

No inference, CUDA imports, extra sampling, timing adjustment or result rewriting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from specrhythm.serving.eager_evidence_export import (
    Reader,
    bounds,
    device_summary,
    duration,
    host_summary,
    intersect,
    interval,
    require,
    touching,
)

MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_STEPS = 512


def analyze(runtime, backend, service, light):
    start, end = (light['measurement_snapshot'][k] for k in (
        'measurement_start_ns', 'measurement_end_ns'))
    require(type(start) is int and type(end) is int and end > start, 'invalid window')
    steps = [s for s in runtime['target_steps'] if s.get('window') and touching(s, start, end)]
    require(0 < len(steps) <= MAX_STEPS, 'missing steps or bounded step limit exceeded')
    require(len(steps) == light['target_steps'], 'runtime step count differs from measured result')
    targets = [d['device'] for d in runtime['target_devices']]
    draft = backend['fixed_device']
    ids = [d['identity'] for d in targets]
    require(len(ids) == 2 and {d['global_rank'] for d in ids} == {0, 1}
            and all(d.get('gpu_uuid') for d in ids)
            and draft['identity'].get('gpu_uuid')
            and len({d['gpu_uuid'] for d in ids} | {draft['identity']['gpu_uuid']}) == 3,
            'invalid within-point TP/device identities')
    forwards = draft['forwards']
    eager = [r for r in forwards if r.get('purpose') == 'eager']
    target_rows = [r for d in targets for r in d['forwards']]
    require(all(type(r.get('gpu_event_ms')) in (int, float)
                and math.isfinite(r['gpu_event_ms']) and r['gpu_event_ms'] > 0
                for r in forwards + target_rows), 'invalid native device duration')
    device = device_summary(draft, start, end, 100000)
    require(device['physical_window_status'] == 'COMPLETE', 'incomplete native Draft bounds')
    overlap = {name: duration(intersect(bounds(target_rows, start, end, inner),
                                       bounds(eager, start, end, inner)))
               for name, inner in (('lower_ms', True), ('upper_ms', False))} if eager else None
    host = backend['fixed_host']['intervals']
    host_window = [r for r in host if touching(r, start, end)]
    proposals = backend.get('fixed_proposals', [])
    cycles = []
    for index, step in enumerate(steps):
        first, last = max(start, step['start_ns']), min(end, step['end_ns'])
        cpu = host_summary(host_window, first, last, 100000)
        launches = [r for r in forwards if first <= r['host_start_ns'] <= last]
        eager_launches = [r for r in launches if r.get('purpose') == 'eager']
        initial = min(eager_launches, key=lambda r: r['host_start_ns']) if eager_launches else None
        enqueue = [r for r in service if r.get('operation') == 'eager_enqueue'
                   and first <= r['received_ns'] <= last]
        latency = {'status': 'NOT_OBSERVED', 'reason': 'no eager forward in step'}
        before = None
        if initial is not None:
            before = host_summary(host_window, first, initial['host_start_ns'], 100000)
            latency = {'status': 'MISSING', 'reason': 'need one unique enqueue RPC in step'}
            if len(enqueue) == 1:
                # Queue.put happens inside service receive -> response completion.
                # No host endpoint is substituted for a GPU execution timestamp.
                q = enqueue[0]
                latency = {
                    'status': 'BOUNDED_FROM_SERVICE_RPC',
                    'lower_ms': (initial['start_lower_ns']-q['completed_ns']) / 1e6,
                    'upper_ms': (initial['start_upper_ns']-q['received_ns']) / 1e6,
                    'enqueue_interval_ns': [q['received_ns'], q['completed_ns']],
                    'first_gpu_start_bounds_ns': [initial['start_lower_ns'],
                                                  initial['start_upper_ns']],
                    'correlation': 'unique enqueue RPC and first eager forward in same step',
                }
        repair = [r for r in launches if r.get('purpose') == 'commit']
        normal = [r for r in proposals if touching(r, first, last)]
        cycles.append({
            'step_index': index, 'request_ids': step['request_ids'],
            'boundary_ns': [first, last], 'step_wall_ms': (last-first) / 1e6,
            'committed_tokens': step.get('committed_tokens'),
            'host': cpu, 'before_first_eager_host_launch': before,
            'enqueue_to_first_gpu': latency,
            'draft_forwards': device_summary({'forwards': launches}, first, last, 100000)[
                'by_purpose'],
            'repair_gpu_union_lower_ms': duration(bounds(repair, first, last, True)),
            'repair_gpu_union_upper_ms': duration(bounds(repair, first, last, False)),
            'normal_recovery_host_union_ms': duration([
                (max(first, interval(r)[0]), min(last, interval(r)[1])) for r in normal]),
            'normal_recovery_request_ids': [r['request_ids'] for r in normal],
        })
    gpu_cover = duration(bounds(forwards, start, end, False))
    host_cover = duration([(max(start, r['start_ns']), min(end, r['end_ns']))
                           for r in host_window])
    joint = duration(bounds(forwards + target_rows, start, end, False) + [
        (max(start, r['start_ns']), min(end, r['end_ns'])) for r in host_window])
    return {
        'original_result': light, 'measurement_boundary_ns': [start, end],
        'throughput_tok_s': light['decode_throughput_tok_s'],
        'committed_tokens_per_step': light['committed_window_tokens'] / len(steps),
        'step_wall_mean_ms': sum(r['step_wall_ms'] for r in cycles) / len(cycles),
        'native_eager_overlap': overlap,
        'overlap_semantics': 'within-point TP union intersected with eager device union',
        'draft_device': device, 'draft_host': host_summary(host_window, start, end, 100000),
        'draft_gpu_outer_union_ms': gpu_cover, 'draft_host_union_ms': host_cover,
        'uncovered_by_draft_host_or_any_recorded_gpu_outer_ms': max(0, (end-start)/1e6-joint),
        'coverage_semantics': 'overlapping/nested coverage; uncovered is not inferred idle time',
        'cycles': cycles,
    }


def report(root, output, expected_commit):
    root, output = Path(root).resolve(), Path(output)
    require(re.fullmatch(r'[0-9a-f]{40}', expected_commit), 'expected full SHA required')
    require(not output.is_symlink() and not output.exists(), 'output collision')
    output = output.resolve()
    require(not output.is_relative_to(root), 'output must be outside the existing result root')
    reader = Reader(root, MAX_FILE_BYTES, MAX_TOTAL_BYTES, 100000)
    config = reader.read(root / 'scan-config.json', required=True)
    require(config['execution']['git_commit'] == expected_commit, 'scan commit mismatch')
    points = {}
    for path in sorted((root / 'runs').glob('*/point.json')):
        point = reader.read(path, required=True)
        if point.get('probe') or point.get('mode') not in ('serial', 'serial-eager'):
            continue
        mode, directory = point['mode'], path.parent
        require(point['batch'] == 16 and point['kind'] == 'decode-scan'
                and point.get('scan') is True and mode not in points, 'ambiguous B16 point')
        light = reader.read(directory / 'light-summary.json', required=True)
        life = reader.read(directory / 'process-lifecycle.json', required=True)
        require(life.get('owned_cleanup_completed') and not life['remaining_owned_pids']
                and type(life.get('exit_monotonic_ns')) is int, 'point has not ended')
        require(light['git_commit'] == expected_commit
                and light['workload_sha256'] == config['workload_sha256'],
                'point identity differs')
        runtime = reader.read(directory / 'runtime.json', required=True)
        backend = reader.read(directory / 'draft-backend-report.json', required=True)
        service = reader.read(directory / 'draft-work-events.jsonl', required=True, jsonl=True)
        name = str((directory / 'draft-work-events.jsonl').relative_to(root))
        require(not reader.inventory[name]['truncated'], 'service evidence exceeds row cap')
        points[mode] = analyze(runtime, backend, service, light)
        # Release raw trees before reading the next point; retain small inventory.
        reader.cache.clear()
    require(set(points) == {'serial', 'serial-eager'}, 'need both ended B16 points')
    value = {'schema_version': 'specrhythm.eager-b16-retest.v1',
             'source_commit': expected_commit, 'workload_sha256': config['workload_sha256'],
             'points': points, 'inventory': list(reader.inventory.values()),
             'reporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'inference_started': False, 'original_results_unchanged': True,
             'limits': {'file_bytes': MAX_FILE_BYTES, 'total_bytes': MAX_TOTAL_BYTES,
                        'steps_per_point': MAX_STEPS}}
    payload = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    require(len(payload) < 64*1024*1024, 'report exceeds 64 MiB output bound')
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as handle:
        handle.write(payload)
    return {'output': str(output), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(report(args.root, args.output, args.expected_commit)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
