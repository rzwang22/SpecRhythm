from __future__ import annotations

import json
import shutil
from copy import deepcopy

import pytest
from test_phase4_d6_comparison import evidence as _evidence
from test_phase4_draft_production_comparison import append_rows, write

from specrhythm.phase4.dual_microbatch import FIELDS
from specrhythm.phase4.dual_microbatch_sweep import (
    SIZES,
    compare_sweep,
    read,
    render_sweep,
    summarize_cell,
)
from specrhythm.phase4.dual_overlap_characterization import characterize_overlap
from specrhythm.phase4.dual_runner import build_cycle_and_overlap_events
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.performance import _validate_mode_boundary


def timing_evidence(overlap=True):
    drafts = []
    for i, (a, b) in enumerate(((10, 20), (30, 40))):
        drafts.append({'request_id': f'd{i}', 'result': {
            'proposal': {'proposal_id': f'p{i}', 'round_id': 0},
            'draft_gpu_interval': {'host_start_ns': a, 'host_end_ns': b,
                'physical_gpu_id': 0, 'cuda_elapsed_ns': b - a,
                'cuda_events': True, 'cuda_synchronized': True}}})
    start, end = (15, 35) if overlap else (50, 60)
    verifies = []
    # Per-request TP event rows naturally have different request IDs and times.
    for rid in ('r0', 'r1'):
        ranks = [{'request_id': rid, 'tp_rank': i, 'global_rank': i, 'physical_gpu_id': i + 1,
                  'gpu_uuid': f'GPU-{i}', 'logical_cuda_index': i,
                  'cuda_events': True, 'cuda_synchronized': True, 'cuda_elapsed_ns': end - start,
                  'host_start_ns': start, 'host_end_ns': end} for i in (0, 1)]
        verifies.append({'request_id': rid, 'verify_microbatch_id': 'v0',
                         'verify_request_ids': ['r0', 'r1'], 'verify_host_start_ns': start,
                         'verify_host_end_ns': end, 'target_rank_intervals': ranks,
                         'target_physical_gpu_ids': [1, 2]})
    _, overlaps = build_cycle_and_overlap_events(drafts, verifies)
    return drafts, verifies, overlaps


@pytest.mark.parametrize('overlap', [False, True])
def test_zero_overlap_valid_actual_intersection_union_excludes_envelope_gaps(overlap):
    result = characterize_overlap(*timing_evidence(overlap))
    assert result['valid'], result['errors']
    assert result['physical_overlap_valid'] is overlap
    assert result['observed_overlap_ms'] == (10 if overlap else 0) / 1e6
    assert result['overlap_is_critical_path_time_saved'] is False


@pytest.mark.parametrize('mutation', ['missing', 'draft-time', 'target-time', 'cuda', 'uuid',
                                    'aliased-device', 'invented-overlap', 'bad-cohort'])
@pytest.mark.parametrize('overlap', [False, True])
def test_bad_timing_or_unsupported_claim_blocked_even_with_valid_witness(overlap, mutation):
    d, v, o = timing_evidence(overlap)
    if mutation == 'missing':
        d = []
    elif mutation == 'draft-time':
        d[1]['result']['draft_gpu_interval']['host_end_ns'] = 0
    elif mutation == 'target-time':
        v[1]['verify_host_end_ns'] = -1
    elif mutation == 'cuda':
        v[1]['target_rank_intervals'][0]['cuda_synchronized'] = False
    elif mutation == 'uuid':
        v[1]['target_rank_intervals'][0]['gpu_uuid'] = ''
    elif mutation == 'aliased-device':
        v[1]['target_rank_intervals'][1]['physical_gpu_id'] = 1
    elif mutation == 'invented-overlap':
        o[0]['overlap_duration_ns'] += 1
    else:
        v[1]['verify_request_ids'] = ['r1']
    assert not characterize_overlap(d, v, o)['valid']


def test_informational_overlap_metadata_does_not_block():
    d, v, o = timing_evidence()
    o[0].pop('schema_version')
    o[0]['overlap_ratio_for_diagnostic_only'] = None
    assert characterize_overlap(d, v, o)['valid']


def test_measurement_characterization_zero_overlap_default_gate_preserved(tmp_path):
    d, v, o = timing_evidence(False)
    for name, rows in (('draft-work-events.jsonl', d), ('verification-events.jsonl', v),
                       ('overlap-events.jsonl', o), ('proposal-lifecycle-events.jsonl',
                       [{'lifecycle_state': 'CREATED', 'draft_start_ns': 10}])):
        append_rows(tmp_path / name, rows)
    raw = {'runtime_semantics': {'overlap_requirement': 'characterization'}}
    assert not _validate_mode_boundary('dual-batch', tmp_path, raw, {}, 1)
    raw['runtime_semantics']['overlap_requirement'] = 'required'
    assert _validate_mode_boundary('dual-batch', tmp_path, raw, {}, 1)


evidence = _evidence


@pytest.fixture
def sweep(evidence):
    root, workload = evidence
    workload.write_text('\n'.join(json.dumps({**json.loads(r), 'maximum_new_tokens': 16})
                                  for r in workload.read_text().splitlines()))
    for mode in ('target', 'serial', 'dual'):
        directory = root / mode
        perf = read(directory / 'decode-performance.json')
        perf['workload_sha256'] = sha256_file(workload)
        perf['metrics']['total_measured_committed_output_tokens'] = 1487
        for i, row in enumerate(perf['requests']):
            ids = [21] * (15 if i < 87 else 14)
            row.update(measured_committed_output_token_ids=ids,
                       measured_committed_output_token_count=len(ids),
                       total_generated_token_ids=[row['bootstrap_token_id'], *ids],
                       finish_reason='length' if i < 87 else 'stop')
        write(directory / 'decode-performance.json', perf)
    for n in SIZES:
        directory = root / f'dual-mb{n}'
        shutil.copytree(root / 'dual', directory)
        d, v, o = timing_evidence(n == 2)
        for name, rows in (('draft-work-events.jsonl', d), ('verification-events.jsonl', v),
                           ('overlap-events.jsonl', o), ('scheduler-events.jsonl',
                           [{**dict.fromkeys(FIELDS, n), 'verify_request_ids': ['r0', 'r1']}])):
            append_rows(directory / name, rows)
        raw = read(directory / 'resident-dual.json')
        raw.update(dict.fromkeys(FIELDS, n))
        raw['runtime_semantics'] = {'overlap_requirement': 'characterization'}
        raw['overlap_gate'] = {'valid': n == 2}
        write(directory / 'resident-dual.json', raw)
        for name in ('decode-performance.json', 'plugin-report.json', 'runtime-manifest.json'):
            p = directory / name
            value = read(p) if p.exists() else {}
            value.update(dict.fromkeys(FIELDS, n))
            if name == 'decode-performance.json':
                value['artifact_sha256']['raw_run'] = sha256_file(directory / 'resident-dual.json')
            write(p, value)
    for name in ('target', 'serial', *(f'dual-mb{n}' for n in SIZES)):
        mode = 'dual' if name.startswith('dual') else name
        n = int(name[7:]) if mode == 'dual' else None
        result = summarize_cell(root / name, mode, workload, n)
        assert result['valid'], result['errors']
        write(root / name / 'qualification.json', result)
    identity = result['experiment_identity']
    write(root / 'preflight.json', {'valid': True, 'errors': [],
        'execution_git_commit': identity['execution_git_commit'],
        'config_sha256': identity['config_sha256'],
        'inputs': {'SR_PHASE4B_WORKLOAD': {'sha256': sha256_file(workload)}}})
    return root, workload


def test_full_sweep_zero_overlap_and_slow_cell_valid(sweep):
    root, workload = sweep
    report = compare_sweep(root)
    assert report['valid'], report['errors']
    assert 'Metric | 2 | 4 | 8 | 16 | 32 | 64 | 100' in render_sweep(report)
    assert len(report['cells']) == 9
    assert report['cells']['dual-mb100']['metrics']['observed_overlap_ms'] == 0
    path = root / 'dual-mb100' / 'decode-performance.json'
    p = read(path)
    p['metrics']['aggregate_throughput_tokens_per_second'] = 0.001
    p['metrics']['decode_makespan_ms'] = 1e12
    write(path, p)
    cell = summarize_cell(path.parent, 'dual', workload, 100)
    assert cell['valid'], cell['errors']
    write(path.parent / 'qualification.json', cell)
    assert compare_sweep(root)['valid']


@pytest.mark.parametrize('artifact', ['resident-dual.json', 'runtime-manifest.json',
                                     'plugin-report.json', 'decode-performance.json'])
def test_recorded_effective_bound_mismatch_fails(sweep, artifact):
    root, workload = sweep
    directory = root / 'dual-mb4'
    value = read(directory / artifact)
    value[FIELDS[1]] = 2
    write(directory / artifact, value)
    perf = read(directory / 'decode-performance.json')
    perf['artifact_sha256']['raw_run'] = sha256_file(directory / 'resident-dual.json')
    write(directory / 'decode-performance.json', perf)
    result = summarize_cell(directory, 'dual', workload, 4)
    assert not result['valid'] and 'microbatch mismatch' in result['errors'][0]


@pytest.mark.parametrize('mutation', ['tokens', 'request-count', 'raw-invalid', 'cleanup',
                                    'nonempty-errors', 'nan', 'tp-consensus'])
def test_material_validity_still_blocks(sweep, mutation):
    root, workload = sweep
    directory = root / 'dual-mb4'
    p = read(directory / 'decode-performance.json')
    raw = read(directory / 'resident-dual.json')
    if mutation == 'tokens':
        p['requests'][0]['measured_committed_output_token_count'] += 1
    elif mutation == 'request-count':
        p['request_count'] = 99
    elif mutation == 'raw-invalid':
        raw['valid'] = False
    elif mutation == 'cleanup':
        p['cleanup_valid'] = False
    elif mutation == 'nonempty-errors':
        raw['errors'] = ['execution failed']
    elif mutation == 'nan':
        p['metrics']['aggregate_throughput_tokens_per_second'] = float('nan')
    else:
        write(directory / 'plugin-report.json', {**dict.fromkeys(FIELDS, 4),
                                              'sampled_row_tp_consensus': False})
    write(directory / 'resident-dual.json', raw)
    p['artifact_sha256']['raw_run'] = sha256_file(directory / 'resident-dual.json')
    write(directory / 'decode-performance.json', p)
    assert not summarize_cell(directory, 'dual', workload, 4)['valid']


def test_cross_session_mismatch_blocks_but_historical_structure_only_warns(sweep):
    root, _ = sweep
    report = compare_sweep(root)
    assert report['valid'] and report['baseline_warnings']
    path = root / 'dual-mb4' / 'qualification.json'
    value = deepcopy(read(path))
    value['experiment_identity']['execution_git_commit'] = 'other'
    write(path, value)
    assert not compare_sweep(root)['valid']


def test_offline_cli_writes_immutable_common_bound_sidecar(sweep, monkeypatch):
    from specrhythm.phase4.dual_microbatch_sweep import main

    root, workload = sweep
    directory = root / 'dual-mb4'
    output = directory / 'cell-cli.json'
    monkeypatch.setattr('sys.argv', ['sweep', 'validate', '--run-root', str(directory),
        '--mode', 'dual', '--workload', str(workload), '--microbatch-size', '4',
        '--output', str(output)])
    assert main() == 0
    binding = read(directory / 'dual-microbatch.json')
    assert all(binding[k] == 4 for k in FIELDS)
    assert binding['runtime_artifact_sha256']['draft-backend-report.json'] == sha256_file(
        directory / 'draft-backend-report.json')
    with pytest.raises(ValueError, match='fresh'):
        main()


def test_seven_cell_set_and_mb100_endpoint_observations(sweep):
    root, _ = sweep
    assert SIZES == (2, 4, 8, 16, 32, 64, 100)
    path = root / 'dual-mb100' / 'qualification.json'
    cell = read(path)
    cell['metrics']['throughput_tokens_per_second'] = 1e6
    write(path, cell)
    report = compare_sweep(root)
    assert report['valid'], report['errors']
    obs = report['observations']
    assert obs['best_observed_microbatch'] == 100
    assert obs['peak_is_intermediate'] is False
    assert {'mb100_minus_mb2', 'draft_p50_mb2_mb100', 'verify_p50_mb2_mb100',
            'mb100_over_serial_throughput'} <= obs.keys()
    assert not any('mb64' in key for key in obs)
    assert all(report['cells']['dual-mb100'][key] == 100 for key in FIELDS)
    header = next(line for line in render_sweep(report).splitlines()
                  if line.startswith('Metric |'))
    assert len(header.split('|')) == 8
    path = root / 'dual-mb64' / 'qualification.json'
    cell = read(path)
    cell['metrics']['throughput_tokens_per_second'] = 2e6
    write(path, cell)
    obs = compare_sweep(root)['observations']
    assert obs['best_observed_microbatch'] == 64 and obs['peak_is_intermediate'] is True
