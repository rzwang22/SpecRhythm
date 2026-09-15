"""Repair attribution uses actual windows, GPU bounds and nested host unions."""

import json

import pytest
from test_eager_evidence_export import NS, SOURCE_COMMIT, write
from test_eager_evidence_export import existing as _existing

from specrhythm.serving.eager_retest_report import analyze, report

existing = _existing


@pytest.fixture
def inputs(existing):
    root, output = existing
    config = json.loads((root / 'scan-config.json').read_text())
    config['workload_sha256'] = 'same-workload'
    write(root / 'scan-config.json', config)
    for path in (root / 'runs').iterdir():
        light = json.loads((path / 'light-summary.json').read_text())
        light.update(workload_sha256='same-workload', decode_throughput_tok_s=100,
                     committed_window_tokens=30, target_steps=3)
        write(path / 'light-summary.json', light)
        runtime = json.loads((path / 'runtime.json').read_text())
        for step in runtime['target_steps']:
            step.update(request_ids=['r'], committed_tokens=10)
        write(path / 'runtime.json', runtime)
        backend = json.loads((path / 'draft-backend-report.json').read_text())
        backend['fixed_host']['intervals'] = [
            {'category': 'prefix_hash_and_block_record', 'start_ns': 100*NS, 'end_ns': 110*NS},
            {'category': 'json_serialization', 'start_ns': 102*NS, 'end_ns': 108*NS},
            {'category': 'resident_block_audit', 'start_ns': 110*NS, 'end_ns': 115*NS},
            {'category': 'resident_block_audit', 'start_ns': 1*NS, 'end_ns': 99*NS},
        ]
        write(path / 'draft-backend-report.json', backend)
    return root, output


def test_report_uses_window_union_and_bounded_enqueue_to_gpu_latency(inputs):
    root, output = inputs
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    report(root, output, SOURCE_COMMIT)
    result = json.loads(output.read_text())
    assert result['original_results_unchanged'] and not result['inference_started']
    point = result['points']['serial-eager']
    assert point['native_eager_overlap'] == {'lower_ms': 10, 'upper_ms': 10}
    assert point['committed_tokens_per_step'] == 10 and point['step_wall_mean_ms'] == 100
    assert point['draft_host_union_ms'] == 15  # JSON is nested; setup is excluded.
    assert point['cycles'][0]['host']['by_category']['resident_block_audit']['count'] == 1
    latency = point['cycles'][0]['enqueue_to_first_gpu']
    assert latency['status'] == 'BOUNDED_FROM_SERVICE_RPC'
    assert latency['lower_ms'] == 4 and latency['upper_ms'] == 5
    assert point['cycles'][1]['enqueue_to_first_gpu']['status'] == 'NOT_OBSERVED'
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_unknown_or_ambiguous_enqueue_is_not_an_exact_latency(inputs):
    root, _ = inputs
    directory = root / 'runs/actual-ended-serial-eager'
    runtime, backend, light = [json.loads((directory / name).read_text()) for name in (
        'runtime.json', 'draft-backend-report.json', 'light-summary.json')]
    result = analyze(runtime, backend, [], light)
    assert result['cycles'][0]['enqueue_to_first_gpu']['status'] == 'MISSING'
    service = {'operation': 'eager_enqueue', 'received_ns': 115*NS, 'completed_ns': 116*NS}
    result = analyze(runtime, backend, [service, service], light)
    assert result['cycles'][0]['enqueue_to_first_gpu']['status'] == 'MISSING'


def test_retest_report_rejects_wrong_commit_and_invalid_native_time(inputs):
    root, output = inputs
    with pytest.raises(ValueError, match='commit mismatch'):
        report(root, output, 'f'*40)
    path = root / 'runs/actual-ended-serial-eager/draft-backend-report.json'
    backend = json.loads(path.read_text())
    backend['fixed_device']['forwards'][0]['gpu_event_ms'] = -1
    write(path, backend)
    with pytest.raises(ValueError, match='native device duration'):
        report(root, output, SOURCE_COMMIT)
    assert not output.exists()
