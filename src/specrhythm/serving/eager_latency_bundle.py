"""Bounded offline evidence archive, including failure-time native/host sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

FILE_LIMIT = 512*1024*1024
TOTAL_LIMIT = 1024*1024*1024


def bundle(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(root):
        raise ValueError('new archive outside root required')
    candidates = [(root.parent / (root.name+suffix), 'summary/'+root.name+suffix)
                  for suffix in ('-complete-small.tar.gz', '-failure-small.tar.gz',
                                 '-operator-small.tar.gz', '-manual-small.tar.gz',
                                 '-attribution.json', '-timeline.svg')]
    for point in sorted((root/'runs').glob('*/point.json')):
        if (point.is_symlink() or not point.resolve().is_relative_to(root)
                or point.stat().st_size > 65536):
            raise ValueError('invalid bounded point metadata')
        value = json.loads(point.read_text())
        if value.get('probe') or value.get('mode') not in ('serial', 'serial-eager'):
            continue
        if value.get('batch') != 16:
            raise ValueError('latency evidence is B16 only')
        for name in ('point.json', 'runtime.json', 'draft-backend-report.json',
                     'draft-work-events.jsonl', 'process-lifecycle.json'):
            p = point.parent/name
            candidates.append((p, 'raw/'+str(p.relative_to(root))))
    if len(candidates) > 32:
        raise ValueError('bounded point/file count exceeded')
    inventory, total = [], 0
    with output.open('xb') as handle, tarfile.open(fileobj=handle, mode='w:gz') as archive:
        for source, name in candidates:
            row = {'path': name}
            inventory.append(row)
            if not source.exists():
                row['status'] = 'MISSING'
                continue
            if source.is_symlink() or not source.is_file():
                row['status'] = 'OMITTED_NONREGULAR'
                continue
            before = source.stat()
            row['source_bytes'] = before.st_size
            if before.st_size > FILE_LIMIT or total+before.st_size > TOTAL_LIMIT:
                row['status'] = 'OMITTED_LIMIT'
                continue
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                remaining_bytes = before.st_size
                while remaining_bytes:
                    block = stream.read(min(1024*1024, remaining_bytes))
                    if not block:
                        raise ValueError('source shortened during bounded export')
                    digest.update(block)
                    remaining_bytes -= len(block)
                stream.seek(0)
                info = tarfile.TarInfo(name)
                info.size, info.mtime = before.st_size, int(before.st_mtime)
                archive.addfile(info, stream)
            after = source.stat()
            row.update(status='INCLUDED', sha256_before_archive=digest.hexdigest(),
                       stable=(before.st_size, before.st_mtime_ns, before.st_ino) == (
                           after.st_size, after.st_mtime_ns, after.st_ino))
            total += before.st_size
        payload = json.dumps({'schema_version': 'specrhythm.eager-latency-bundle.v1',
                              'inventory': inventory, 'source_bytes_included': total,
                              'limits': {'file_bytes': FILE_LIMIT, 'total_bytes': TOTAL_LIMIT},
                              'source_results_unchanged': True,
                              'semantics': 'MISSING/OMITTED or unstable sources are explicit; '
                              'archive existence does not establish qualification'},
                             indent=2).encode()
        info = tarfile.TarInfo('evidence-inventory.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return {'output': str(output), 'bytes': output.stat().st_size, 'inventory': inventory}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(bundle(args.root, args.output)))


if __name__ == '__main__':
    main()
