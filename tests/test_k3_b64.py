"""Explicit B64 through real capacity/run/report/entry and native producer hooks."""

import copy
import json
import re
import tarfile
from pathlib import Path

import pytest
from test_k3_acceptance import entry, retain
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results, fixed_results
from specrhythm.serving.common import DataError
from specrhythm.serving.decode_scan_plan import selected_point
from specrhythm.serving.k3 import B64, MODES, configuration_of, geometry, matches_geometry
from specrhythm.serving.k3_acceptance import full_batch_receipt, native_geometry
from specrhythm.serving.k3_capacity import qualify
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('stage', ['capacity_probe', 'correctness', 'performance'])
def test_B64_real_run_capacity_summary_export(mode, stage, driven, tmp_path):
    h = driven(mode, stage, full_run=True, configuration=B64)
    summary = fixed_results.summarize if stage == 'correctness' else decode_scan_results.summarize
    result = summary(h.path, h.directory, h.point, probe=h.probe)
    assert result['valid'], result.get('primary_error')
    assert h.runtime['point']['k3_configuration'] == B64
    assert h.runtime['capacity']['active_request_limit'] == 64
    assert h.runtime['capacity']['warmup_rotation'].startswith('64 completed')
    assert qualify(h.actual, mode)['status'] == 'PASS'
    if not h.probe:
        proof = native_geometry(h.runtime, mode, full_fixture=True, configuration=B64)
        assert full_batch_receipt(proof, mode, B64)
        assert not full_batch_receipt(proof, mode)  # no B64 interpretation without declaration
        assert len(proof['first_full_batch']['request_ids']) == (64 if mode in MODES[:2] else 32)
    if stage == 'performance':
        root, result = retain(h)
        body = re.search(r"<<'PY_MEASUREMENT'\n(.*?)\nPY_MEASUREMENT",
                         Path('scripts/run_k3_b64.sh').read_text(), re.S)[1]
        accepted = entry(root, mode, body)
        assert accepted.returncode == 0, accepted.stderr
        assert result['scan_warmup_boundary']['request_opportunities'] == 128
        assert result['request_verification_opportunities'] == sum(
            s['B'] for s in h.runtime['target_steps'] if s['window'])
    archive = tmp_path / 'one.tar.gz'
    export(h.directory, archive, first_code=1, modes=MODES)
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile('inventory.json'))['logical_paths']
        runtime = json.load(t.extractfile(paths['runtime.json']))
        actual = json.load(t.extractfile(paths['actual-capacity.json']))
    assert qualify(actual, mode) == qualify(h.actual, mode)
    if not h.probe:
        assert native_geometry(runtime, mode, configuration=B64) == native_geometry(
            h.runtime, mode, configuration=B64)


@pytest.mark.parametrize('mode', MODES)
def test_B64_explicit_point_and_geometry(mode):
    assert geometry(mode)['active_limit'] == 16
    g = geometry(mode, B64)
    assert g['active_limit'] == g['draft_physical_batch_ceiling'] == 64
    assert g['home_capacities'] == ({'A': 64} if mode in MODES[:2] else {'A': 32, 'B': 32})
    assert matches_geometry(g, mode, B64) and not matches_geometry(g, mode)
    assert configuration_of(selected_point(mode, 64, k3_configuration=B64)) == B64
    with pytest.raises(DataError):
        selected_point(mode, 64)
    with pytest.raises(DataError):
        selected_point(mode, 16, k3_configuration=B64)


@pytest.mark.parametrize('fault', ['missing_config', 'missing_rank', 'duplicate', 'split',
                                 'geometry', 'query', 'sequence', 'runtime_active', 'home_limit'])
def test_B64_raw_evidence_fail_closed(fault, driven):
    h = driven(MODES[0], 'correctness', full_run=True, configuration=B64)
    r, a = copy.deepcopy(h.runtime), copy.deepcopy(h.actual)
    if fault in ('query', 'sequence'):
        key = 'max_num_batched_tokens' if fault == 'query' else 'max_num_seqs'
        a['target_effective_by_rank'][0][key] = 16
        with pytest.raises(DataError):
            qualify(a, MODES[0])
        return
    if fault == 'missing_config':
        del r['point']['k3_configuration']
    elif fault == 'missing_rank':
        r['target_devices'].pop()
    elif fault == 'geometry':
        r['capacity']['execution_geometry'] = geometry(MODES[0])
    elif fault == 'runtime_active':
        r['capacity']['active_request_limit'] = 16
    elif fault == 'home_limit':
        r['capacity']['per_cohort_capacity'] = 32
    else:
        f = r['target_devices'][0]['device']['forwards'][0]
        if fault == 'duplicate':
            f['internal_request_ids'][0] = f['internal_request_ids'][1]
        else:
            r['target_devices'][0]['device']['forwards'].append(copy.deepcopy(f))
    with pytest.raises(DataError):
        native_geometry(r, MODES[0], full_fixture=True, configuration=B64)


@pytest.mark.parametrize('limit', [16, 32])
def test_hidden_serial_subbatches_never_satisfy_B64(driven, limit):
    h = driven(MODES[0], 'correctness', full_run=True, configuration=B64, admission_limit=limit)
    with pytest.raises(DataError, match='lacks one native Target forward'):
        native_geometry(h.runtime, MODES[0], full_fixture=True, configuration=B64)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('ceiling', [1, 7, 16, 32])
def test_warmup_actual_partial_opportunities(mode, ceiling):
    from specrhythm.serving.k3_window import K3Window

    w = K3Window(dict(warmup_steps=2, window_seconds=30), 64, mode, B64)
    count = 0
    while count < 128:
        n = min(ceiling, 128 - count)
        w.step_completed(dict(B=n, request_ids=[str(j) for j in range(n)], cohort='A',
                              start_ns=count+1, committed_tokens=n, rows=[]), count+2)
        count += n
        assert w.warmup_rotations == count // 64
    assert w.warmup_boundary()['request_opportunities'] == 128
    assert w.warmup_boundary()['completed_steps'] == (128 + ceiling - 1) // ceiling


@pytest.mark.parametrize('mode', MODES)
def test_B64_capacity_real_query_KV_deficit_and_config(mode):
    from types import SimpleNamespace as NS

    from specrhythm.serving.fixed_plan import capacity_metadata
    from specrhythm.serving.k3_capacity import check

    meta = capacity_metadata(mode, active_limit=64, resident_requirement=360,
                             target_sequence_limit=512, k3_configuration=B64)
    defs = [NS(request_id=str(i), prompt_length=15, maximum_new_tokens=100) for i in range(360)]
    for role, gpu in [('target', 1), ('draft', 0)]:
        rank = dict(role=role, mode=mode, physical_gpu_id=gpu, gpu_uuid=f'GPU-{gpu}',
                    block_size=16, num_gpu_blocks=10000, vocab_size=100, free_memory_bytes=2**30)
        r = check(defs, rank, mode=mode, active_limit=64, metadata=meta)
        assert r['active_limit'] == 64 and r['valid']
        assert r['reserved_speculative_positions'] == (
            6 if role == 'draft' and mode in (MODES[1], MODES[3]) else 4)
        bad = check(defs, dict(rank, num_gpu_blocks=400), mode=mode,
                    active_limit=64, metadata=meta)
        assert not bad['valid'] and bad['required_blocks'] > bad['num_gpu_blocks']
        with pytest.raises(DataError, match='batch/query'):
            check(defs, rank, mode=mode, active_limit=64,
                  metadata={**meta, role+'_query_token_limit': 31})


@pytest.mark.parametrize('fault', ['manifest', 'point', 'capacity', 'ceiling', 'mode'])
def test_B64_entry_manifest_consistency_before_execution(fault):
    from specrhythm.serving.decode_scan_plan import manifest, options
    from specrhythm.serving.k3 import validate_point

    m = manifest({}, [str(i) for i in range(360)], 'sha', options(), 64, k3_configuration=B64)
    point = selected_point(MODES[0], 64, k3_configuration=B64)
    validate_point(point, m)
    if fault == 'manifest':
        m.pop('k3_configuration')
    elif fault == 'point':
        point['batch'] = 16
    elif fault == 'capacity':
        m['fixed_diagnostic']['capacity'][MODES[0]].pop('k3_configuration')
    elif fault == 'ceiling':
        m['fixed_diagnostic']['capacity'][MODES[0]]['max_requests_per_target_forward'] = 32
    else:
        point['runtime_mode'] = MODES[2]
    with pytest.raises(DataError):
        validate_point(point, m)


def test_B64_actual_joint_fixture_producer_and_output_comparison(tmp_path):
    from test_serving_s1 import request

    from specrhythm.serving.decode_scan_plan import manifest, options
    from specrhythm.serving.prepost_gpu_check import compare_outputs, prepare
    from specrhythm.serving.s2_plan import check_seal

    source = tmp_path / 'source'
    (source / 'inputs').mkdir(parents=True)
    for name in ('config.json', 'patch-manifest.json', 'environment.json', 'topology.json'):
        (source / name).write_text('{}')
    rows = [request(i, 100).to_dict() for i in range(360)]
    raw = '\n'.join(json.dumps(r) for r in rows)
    (source / 'inputs/requests.jsonl').write_text(raw)
    base = manifest({'capacity': {}}, [r['request_id'] for r in rows], 'f'*64,
                    options(), 64, k3_configuration=B64)
    (source / 'inputs/execution-B64.json').write_text(json.dumps(base))
    for mode in ('target', *MODES):
        path = prepare(source, tmp_path / mode, mode, modes=MODES, configuration=B64)
        m = json.loads(path.read_text())
        check_seal(m)
        assert m['active_limit'] == m['actual_N'] == m['requested_N'] == 64
        assert m['k3_configuration'] == B64
        assert m['request_ids'] == [r['request_id'] for r in rows[:64]]
        fixture = [json.loads(r) for r in (path.parent/'requests.jsonl').read_text().splitlines()]
        assert len(fixture) == 64 and {r['maximum_new_tokens'] for r in fixture} == {32}
    assert (source/'inputs/requests.jsonl').read_text() == raw
    value = dict(stop_reason='all_naturally_completed', target_steps=[], requests=[
        dict(request_id=str(i), generated_token_ids=list(range(32 if i else 2)),
             state='FINISHED', resources_released=True, finish_reason='length' if i else 'stop')
        for i in range(64)])
    runs = {m: copy.deepcopy(value) for m in ('target', *MODES)}
    passed = compare_outputs(runs, modes=MODES, require_mixed=False, request_count=64,
                             compare_termination=True)
    assert passed['valid'] and len(passed['comparisons']) == 256
    runs[MODES[0]]['requests'][0]['finish_reason'] = 'length'
    with pytest.raises(DataError, match='termination reasons'):
        compare_outputs(runs, modes=MODES, require_mixed=False, request_count=64,
                        compare_termination=True)


@pytest.mark.parametrize('value', [None, True, 64, 64.0, 'B64', '', 'k3-b32-v1'])
def test_invalid_configuration_never_becomes_B64(value):
    with pytest.raises(DataError):
        geometry(MODES[0], value)


def test_B64_bounded_export_capacity_is_explicit_and_corrupt_receipt_kept(tmp_path):
    from specrhythm.serving.k3_capacity import preflight
    from specrhythm.serving.ping_prepost_delivery import FILE_LIMIT, TOTAL_LIMIT, byte_limits

    assert byte_limits(tmp_path) == (FILE_LIMIT, TOTAL_LIMIT)
    path = tmp_path/'k3-capacity-contract.json'
    path.write_text(json.dumps(preflight(B64)))
    assert byte_limits(tmp_path) == (4*FILE_LIMIT, 4*TOTAL_LIMIT)
    archive = tmp_path/'expanded.tar.gz'
    export(tmp_path, archive, first_code=23, modes=MODES)
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile('inventory.json'))
        assert inv['limits']['unique_payload_bytes'] == 4*TOTAL_LIMIT
        assert inv['first_exit_code'] == 23
    archive.unlink()
    path.write_bytes(b'{"truncated":')
    export(tmp_path, archive, first_code=23, modes=MODES)
    with tarfile.open(archive) as t:
        inv = json.load(t.extractfile('inventory.json'))
        assert inv['first_exit_code'] == 23 and inv['evidence_errors']
        assert t.extractfile(inv['logical_paths'][path.name]).read() == path.read_bytes()
