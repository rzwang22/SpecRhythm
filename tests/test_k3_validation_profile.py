"""Real manifest/drive/qualification/export paths; GPU substitution stays in fixture."""

import copy
import json
import re
import tarfile
from pathlib import Path

import pytest
from test_audit_layer_report import existing as _existing
from test_audit_layer_report import inputs as _inputs
from test_k3_acceptance import entry, retain
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results, fixed_results
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.k3 import B16, B64, MODES, validate_point
from specrhythm.serving.k3_acceptance import measurement, native_geometry
from specrhythm.serving.k3_validation import (
    EXPLORATION,
    STRICT,
    fields,
    matching,
    plan,
    profile_of,
)
from specrhythm.serving.ping_prepost_delivery import export

hardware, produced, driven = _hardware, _produced, _driven
existing, inputs = _existing, _inputs


@pytest.mark.parametrize('mode', MODES)
def test_exploration_real_runtime_entry_qualification_archive(mode, driven, tmp_path):
    h = driven(mode, 'performance', full_run=True, configuration=B64,
               validation_profile=EXPLORATION)
    m = read_json(h.path)
    validate_point(h.point, m)
    root, result = retain(h)
    assert matching(m, result, h.runtime, h.point) == EXPLORATION
    assert result['measurement_valid'] is True
    assert result['formal_comparison_eligible'] is True  # Run-only legacy meaning.
    assert result['output_equivalence_status'] == 'NOT_RUN'
    assert result['full_output_comparison_run'] is False
    assert result['native_geometry_status'] == 'PASS'
    assert result['scan_warmup_boundary']['request_opportunities'] == 128
    body = re.search(r"<<'PY_MEASUREMENT'\n(.*?)\nPY_MEASUREMENT",
                     Path('scripts/run_k3_b64.sh').read_text(), re.S)[1]
    actual = entry(root, mode, body, EXPLORATION)
    assert actual.returncode == 0, actual.stderr
    p = root/'validation-plan.json'
    p.write_text(json.dumps(plan(EXPLORATION, B64)))
    archive = tmp_path/'one.tar.gz'
    inventory = export(root, archive, first_code=23, modes=MODES)
    assert inventory['intentionally_not_run'][0]['path'] == 'joint/'
    assert not any(r['path'].startswith('joint/') for r in inventory['inventory'])
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile('inventory.json'))['logical_paths']
        runtime = json.load(t.extractfile(paths['runs/actual-drive/runtime.json']))
    assert runtime['validation_profile'] == EXPLORATION
    assert measurement(result, runtime, mode, B64, EXPLORATION) == native_geometry(
        h.runtime, mode, full_fixture=True)
    # Intentional absence only applies to the independent joint directory.
    bad = export(root, tmp_path/'incomplete.tar.gz', first_code=0, modes=MODES)
    assert bad['export_validation_exit_code'] == 41
    assert any('required performance evidence missing' in e for e in bad['evidence_errors'])


@pytest.mark.parametrize('fault', ['profile', 'missing_profile', 'rank', 'frontier', 'release'])
def test_exploration_rejects_real_required_evidence(fault, driven):
    h = driven(MODES[0], 'performance', full_run=True, configuration=B64,
               validation_profile=EXPLORATION)
    r = read_json(h.directory/'runtime.json')
    if fault == 'profile':
        r['validation_profile'] = STRICT
    elif fault == 'missing_profile':
        del r['validation_profile']
    elif fault == 'rank':
        r['target_devices'].pop()
    elif fault == 'frontier':
        r['target_steps'][0]['rows'][0]['position_start'] = -1
    else:
        r['requests'][0]['resources_released'] = False
    (h.directory/'runtime.json').write_text(json.dumps(r))
    result = decode_scan_results.summarize(h.path, h.directory, h.point)
    assert result['valid'] is False


@pytest.mark.parametrize('mode', MODES)
def test_exploration_requires_full_native_batch_in_current_run(mode, driven):
    h = driven(mode, 'performance', full_run=True, configuration=B64,
               validation_profile=EXPLORATION, admission_limit=16)
    result = decode_scan_results.summarize(h.path, h.directory, h.point)
    assert not result['valid'] and 'lacks one native Target forward' in result['errors'][0]


@pytest.mark.parametrize('value', [None, True, 1, '', 'performance', []])
def test_invalid_explicit_policy_never_skips_joint(value):
    with pytest.raises(DataError):
        profile_of(dict(validation_profile=value, k3_configuration=B64))
    assert profile_of({}) == STRICT
    assert fields(None, B16) == {}
    with pytest.raises(DataError):
        fields(EXPLORATION, B16)


def test_plan_to_manifest_and_point_is_not_only_an_outer_shell_flag():
    from specrhythm.serving.decode_scan_plan import manifest, options, selected_point

    p = selected_point(MODES[0], 64, k3_configuration=B64, validation_profile=EXPLORATION)
    m = manifest({}, [str(i) for i in range(360)], 'sha', options(), 64,
                 k3_configuration=B64, validation_profile=EXPLORATION)
    validate_point(p, m)
    m.pop('validation_profile')
    with pytest.raises(DataError, match='validation_profile'):
        validate_point(p, m)


def test_strict_joint_aggregate_mismatch_uses_actual_modes_and_raw_sources(
    driven, tmp_path, monkeypatch
):
    from specrhythm.serving import fixed_cli, k3_gpu_check, ping_prepost_gpu_check
    from specrhythm.serving.execution_failure import summarize_joint

    # Actual CPU drive, serialization and formal qualification produce the fixtures.
    # The Target-only GPU substitute emits the same reference as the first K3 run.
    hardware_results = {}
    for mode in MODES:
        h = driven(mode, 'correctness', full_run=True, configuration=B64,
                   validation_profile=STRICT)
        result = fixed_results.summarize(h.path, h.directory, h.point)
        assert result['valid']
        result = fixed_results.emit_result(h.directory, result, h.point)
        hardware_results[mode] = (h, result)
    h, result = hardware_results[MODES[0]]
    target = copy.deepcopy(h.runtime)
    ref = tmp_path/'target-reference'
    ref.mkdir()
    (ref/'runtime.json').write_text(json.dumps(target))
    (ref/'draft-backend-report.json').write_text((h.directory/'draft-backend-report.json').read_text())
    (ref/'light-summary.json').write_text(json.dumps({**result, 'effective_exit_code': 0}))
    # Explicit GPU-output fault at the serialized worker boundary, before comparison.
    for mode in MODES[:2]:
        h, result = hardware_results[mode]
        raw = read_json(h.directory/'runtime.json')
        raw['requests'][0]['generated_token_ids'][-1] = 9999
        (h.directory/'runtime.json').write_text(json.dumps(raw))
        (h.directory/'light-summary.json').write_text(json.dumps(result))
    called = []

    def run_point(root, selected, **kw):
        mode = selected['mode']
        called.append(mode)
        if mode == 'target':
            return ref, hardware_results[MODES[0]][1]
        h, result = hardware_results[mode]
        return h.directory, result

    monkeypatch.setattr(fixed_cli, 'run_point', run_point)
    monkeypatch.setattr(ping_prepost_gpu_check, 'prepare', lambda *a, **kw: tmp_path/'unused')
    delivery = tmp_path/'delivery'
    with pytest.raises(DataError, match='full output mismatch'):
        k3_gpu_check.run(tmp_path/'source', delivery/'joint', B64)
    assert called == ['target', *MODES]
    failure = read_json(delivery/'joint/failure.json')
    assert failure['failure_layer'] == 'comparison' and failure['mode'] is None
    assert failure['point'] == 'comparison'
    assert failure['output_equivalence_status'] == failure['output_correctness'] == 'FAILED'
    assert failure['mismatched_modes'] == sorted(MODES[:2])
    assert len(failure['mismatches']) == 2
    assert failure['command_exit_code'] == 1
    assert all(r['effective_exit_code'] == 0 for r in failure['process_results'])
    assert all(r['source'].endswith('/runtime.json') for r in failure['mismatches'])
    first = summarize_joint(delivery/'joint', 1)
    assert first['point'] == 'comparison' and first['primary_error'] == failure['primary_error']
    archive = tmp_path/'strict-failure.tar.gz'
    export(delivery, archive, first_code=1, modes=MODES)
    with tarfile.open(archive) as t:
        i = json.load(t.extractfile('inventory.json'))
        assert json.load(t.extractfile(i['logical_paths']['joint/failure.json'])) == failure
        assert i['first_exit_code'] == 1


def test_comparison_profile_and_export_replay_use_real_native_proofs(
    driven, inputs, tmp_path
):
    """Comparison unit: synthetic host diagnostics + real drive-produced geometry."""
    from test_execution_evidence import qualified

    from specrhythm.serving.execution_evidence import qualify
    from specrhythm.serving.k3_capacity import preflight
    from specrhythm.serving.ping_prepost_delivery import comparison

    d = tmp_path/'comparison-delivery'
    (d/'points').mkdir(parents=True)
    (d/'validation-plan.json').write_text(json.dumps(plan(EXPLORATION, B64)))
    (d/'k3-capacity-contract.json').write_text(json.dumps(preflight(B64)))
    for mode in MODES:
        h = driven(mode, 'performance', full_run=True, configuration=B64,
                   validation_profile=EXPLORATION)
        root, light = retain(h)
        import shutil

        shutil.copytree(root, d/'points'/mode)
        # fixed_cli.run_point publishes this selected point before calling drive.
        # This lower-level fixture enters drive directly; retain its real selection.
        from specrhythm.serving.s1_workload import write_once

        write_once(d/'points'/mode/'runs/actual-drive/point.json', h.point)
        (d/'points'/mode/'scan-config.json').write_text(json.dumps({
            'validation_profile': EXPLORATION, 'k3_configuration': B64}))
        r = qualified(inputs)  # Explicit host/native timing unit fixture; never GPU evidence.
        r.update(mode=mode, **{k: light[k] for k in (
            'validation_profile', 'k3_configuration', 'execution_geometry', 'measurement_valid',
            'output_equivalence_status', 'full_output_comparison_run', 'native_geometry_status',
            'native_target_geometry')}, source_commit='a'*40, workload_sha256='frozen',
            common_execution={'fixture': 'shared'},
            options={'draft_audit': 'runtime', 'observation': 'buffered-live'},
            original_run_details={'capacity_status': 'PASS', 'effective_exit_code': 0},
            pingpong={'status': 'COMPLETE', 'native_overlap': {}, 'pipeline': {
                'dispatch': {'resident_schedule': {'status': 'COMPLETE'}}}})
        for c in r['execution_path']['cycles']:
            c['latencies_ms']['service_receive_to_owner_dequeue'] = 1.0
        assert qualify(r)['diagnostic_integrity'] == 'COMPLETE'
        (d/'points'/f'{mode}-audit-report.json').write_text(json.dumps(r))
        (d/'points'/f'{mode}-evidence-status.json').write_text(json.dumps(qualify(r)))
    result = comparison(d, modes=MODES)
    assert result['valid'] and result['measurement_valid']
    assert result['output_equivalence_status'] == 'NOT_RUN'
    assert result['full_output_comparison_run'] is False
    assert result['native_geometry_source'] == 'current warmup/runtime'
    assert len(result['paired_ratios']) == 4
    out = tmp_path/'success.tar.gz'
    inv = export(d, out, modes=MODES)
    assert inv['missing'] == 0 and inv['export_validation_exit_code'] == 0
    restored = tmp_path/'restored'
    with tarfile.open(out) as t:
        paths = json.load(t.extractfile('inventory.json'))['logical_paths']
        for name, obj in paths.items():
            p = restored/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(t.extractfile(obj).read())
    assert comparison(restored, modes=MODES) == result
    broken = restored/'points'/f'{MODES[0]}-audit-report.json'
    original = read_json(broken)
    for key, value in [('output_equivalence_status', 'PASS'), ('native_target_geometry', None),
                       ('measurement_valid', False), ('validation_profile', STRICT)]:
        broken.write_text(json.dumps({**original, key: value}))
        assert not comparison(restored, modes=MODES)['valid']
    broken.write_text(json.dumps(original))
    (restored/'validation-plan.json').unlink()
    assert not comparison(restored, modes=MODES)['valid']  # Legacy never implicitly opts in.


@pytest.mark.parametrize('profile,configuration,error', [
    (EXPLORATION, B64, 'source SHA differs'),
    (STRICT, B64, 'validation_profile differs'),
    (EXPLORATION, B16, 'geometry differs'),
])
def test_actual_evidence_cli_accepts_runner_flags_and_checks_records(
    tmp_path, profile, configuration, error
):
    from specrhythm.serving.ping_prepost_evidence import main

    (tmp_path/'scan-config.json').write_text(json.dumps({
        'k3_configuration': B64, 'validation_profile': EXPLORATION,
        'execution': {'git_commit': 'a'*40}}))
    # Wrong source commit intentionally reaches the real report reader for matching
    # policy/geometry. Before the parser repair argparse exited on unknown arguments.
    with pytest.raises(DataError, match=error):
        main(['--root', str(tmp_path), '--commit', 'b'*40,
              '--output', str(tmp_path/'report.json'), '--status', str(tmp_path/'status.json'),
              '--k3-configuration', configuration, '--validation-profile', profile])
    assert not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('value', [[], None, {'schema_version': 'specrhythm.k3-validation-plan.v1',
                                           'full_output_comparison_planned': True}])
def test_corrupt_plan_is_retained_and_never_implicitly_selects_exploration(tmp_path, value):
    (tmp_path/'validation-plan.json').write_text(json.dumps(value))
    archive = tmp_path/'corrupt-plan.tar.gz'
    result = export(tmp_path, archive, modes=MODES, first_code=23)
    assert result['first_exit_code'] == 23 and result['export_validation_exit_code'] == 41
    assert any('invalid validation plan' in e for e in result['evidence_errors'])
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile('inventory.json'))['logical_paths']
        assert json.load(t.extractfile(paths['validation-plan.json'])) == value


def test_scan_summary_retains_policy_and_real_geometry(driven):
    import csv

    from specrhythm.serving.decode_scan_cli import summary
    from specrhythm.serving.decode_scan_plan import SCHEMA
    from specrhythm.serving.s2_plan import sealed
    from specrhythm.serving.s2_pool import publish

    h = driven(MODES[0], 'performance', full_run=True, configuration=B64,
               validation_profile=EXPLORATION)
    root, light = retain(h)
    publish(root/'scan-config.json', sealed(dict(
        schema_version=SCHEMA, validation_profile=EXPLORATION, k3_configuration=B64,
        execution={'git_commit': light['git_commit']}, pool_size=360,
        workload_sha256=light['workload_sha256'], points=[], optional_points=[h.point])))
    result = summary(root)
    row = result['points'][0]
    assert result['validation_profile'] == row['validation_profile'] == EXPLORATION
    assert row['output_equivalence_status'] == 'NOT_RUN' and row['sub_batch'] == 64
    assert row['measurement_valid'] and row['native_geometry_status'] == 'PASS'
    with Path(result['artifact']).with_suffix('.csv').open() as f:
        csv_row = list(csv.DictReader(f))[0]
    assert csv_row['validation_profile'] == EXPLORATION
    assert csv_row['output_equivalence_status'] == 'NOT_RUN'
