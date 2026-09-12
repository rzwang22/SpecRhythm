"""Small offline SVG timelines drawn only from retained native interval bounds."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def render(report):
    panels = []
    for mode, point in report['points'].items():
        cycles = point['cycles']
        selected = {0, len(cycles)//2, len(cycles)-1}
        detailed = point.get('critical_path', {}).get('cycles', [])
        # Include observed adverse cases; never synthesize overlap/waste classes.
        for counter in ('parent_rejections', 'bridge_mismatches'):
            candidate = next((i for i, c in enumerate(detailed) if any(
                e.get('counter_delta', {}).get(counter, 0)
                for e in c.get('host_events', []))), None)
            if candidate is not None:
                selected.add(candidate)
        for i in sorted(selected):
            panels.append((mode, cycles[i], detailed[i] if i < len(detailed) else {}))
    width, panel_height = 1120, 190
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{90+panel_height*len(panels)}" viewBox="0 0 {width} '
           f'{90+panel_height*len(panels)}">', '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" fill="#172b4d">',
           '<text x="24" y="28" font-size="18">Serial-eager: measured native GPU intervals</text>',
           '<text x="24" y="51" font-size="12">Solid = inner bound; pale = outer bound. '
           'Time is relative to complete host step start. '
           'Missing Target data stays missing.</text>']
    for n, (mode, cycle, detail) in enumerate(panels):
        y = 88+n*panel_height
        first, last = cycle['boundary_ns']
        def x(t, a=first, b=last):
            return 155+930*(t-a)/(b-a)
        title = f"{mode}, step {cycle['step_index']}, host wall {cycle['step_wall_ms']:.3f} ms"
        svg.append(f'<text x="24" y="{y}" font-size="14">{html.escape(title)}</text>')
        draft = report['points'][mode]['draft_device']['forwards']
        lanes = [('Target rank 0', detail.get('target_ranks', {}).get('0', {}).get('forwards')),
                 ('Target rank 1', detail.get('target_ranks', {}).get('1', {}).get('forwards')),
                 ('Draft eager', [r for r in draft if r.get('purpose') == 'eager'
                                  and first <= r['host_start_ns'] <= last])]
        for j, (name, rows) in enumerate(lanes):
            ly = y+31+30*j
            svg.append(f'<text x="24" y="{ly+12}" font-size="12">{name}</text>')
            svg.append(f'<path d="M155 {ly+18} H1085" stroke="#dce3ec"/>')
            if rows is None:
                svg.append(f'<text x="168" y="{ly+12}" font-size="12" fill="#687787">'
                           'MISSING raw native bounds; not reconstructed from ZERO summary</text>')
            for row in rows or []:
                color = '#cf6a20' if j == 2 else '#126c93'
                for inner in (False, True):
                    a = max(first, row['start_upper_ns' if inner else 'start_lower_ns'])
                    b = min(last, row['end_lower_ns' if inner else 'end_upper_ns'])
                    if b <= a:
                        continue
                    svg.append(f'<rect x="{x(a):.3f}" y="{ly}" width="{max(.6,x(b)-x(a)):.3f}" '
                               f'height="16" fill="{color}" opacity="{1 if inner else .25}"/>')
        q = cycle.get('enqueue_to_first_gpu', {})
        if 'enqueue_interval_ns' in q:
            a, b = q['enqueue_interval_ns']
            svg.append(f'<rect x="{x(a):.3f}" y="{y+24}" width="{max(1,x(b)-x(a)):.3f}" '
                       'height="92" fill="#7154b8" opacity=".20"/>')
            svg.append(f'<text x="155" y="{y+143}" font-size="12">'
                       f'Enqueue RPC → first GPU: {q["lower_ms"]:.3f}–{q["upper_ms"]:.3f} ms; '
                       'purple = enqueue RPC bracket</text>')
        svg.append(f'<text x="155" y="{y+128}" font-size="11">0 ms</text>')
        svg.append(f'<text x="1000" y="{y+128}" font-size="11">'
                   f'{(last-first)/1e6:.1f} ms</text>')
    return '\n'.join(svg+['</g></svg>'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.stat().st_size > 64*1024*1024:
        parser.error('report exceeds 64 MiB bound')
    value = json.loads(args.report.read_text())
    with args.output.open('x') as handle:
        handle.write(render(value))


if __name__ == '__main__':
    main()
