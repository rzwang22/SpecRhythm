"""Native evidence survives export; omissions cannot become qualified GPU data."""

import json
import tarfile
import xml.etree.ElementTree as ET

from specrhythm.serving import eager_latency_bundle
from specrhythm.serving.eager_latency_render import render


def test_failure_bundle_preserves_raw_sources_and_reports_missing_and_limit(tmp_path, monkeypatch):
    root = tmp_path/'root'
    point = root/'runs'/'p'
    point.mkdir(parents=True)
    (point/'point.json').write_text(json.dumps({'mode': 'serial-eager', 'batch': 16}))
    (point/'runtime.json').write_text('{"native": "retained"}')
    (point/'draft-backend-report.json').write_bytes(b'x'*512)
    monkeypatch.setattr(eager_latency_bundle, 'FILE_LIMIT', 256)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    output = tmp_path/'failure.tar.gz'
    result = eager_latency_bundle.bundle(root, output)
    rows = {r['path']: r for r in result['inventory']}
    assert rows['raw/runs/p/runtime.json']['status'] == 'INCLUDED'
    assert rows['raw/runs/p/draft-backend-report.json']['status'] == 'OMITTED_LIMIT'
    assert rows['raw/runs/p/draft-work-events.jsonl']['status'] == 'MISSING'
    with tarfile.open(output) as archive:
        assert archive.extractfile('raw/runs/p/runtime.json').read() == b'{"native": "retained"}'
        inventory = json.load(archive.extractfile('evidence-inventory.json'))
        assert inventory['source_results_unchanged']
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_svg_never_draws_missing_target_or_invented_overlap():
    point = {'cycles': [{'step_index': 0, 'boundary_ns': [0, 1000000000], 'step_wall_ms': 1000}],
             'draft_device': {'forwards': []}}
    svg = render({'points': {'serial-eager': point}})
    ET.fromstring(svg)
    assert svg.count('MISSING raw native bounds') == 2
    assert 'solid' not in svg.lower().split('missing raw native bounds')[1].split('</text>')[0]


def test_enabled_observation_preserves_eager_disabled_backend_calls(monkeypatch):
    from test_serial_eager_batch_settlement import (
        test_eager_disabled_uses_the_original_batched_commit_and_generation_path,
    )

    from specrhythm.continuation.trace import TRACE

    monkeypatch.setattr(TRACE, 'enabled', True)
    test_eager_disabled_uses_the_original_batched_commit_and_generation_path()
