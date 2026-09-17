"""Real control publication/consumers and drive->qualify->archive, GPU bodies substituted."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.continuation.trace import CausalTrace
from specrhythm.serving import decode_scan_results, fixed_runtime, s2_pool
from specrhythm.serving.k3 import B128, MODES
from specrhythm.serving.k3_repeat_run import declaration
from specrhythm.serving.ping_prepost_delivery import export
from specrhythm.serving.target_dispatch import qualification_errors

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("policy", ["reference", "encode-once"])
def test_real_drive_publish_native_qualify_export(mode, policy, driven, monkeypatch, tmp_path):
    from specrhythm.continuation import trace

    t = CausalTrace(enabled=True, layout="phased")
    monkeypatch.setattr(trace, "TRACE", t)
    h = driven(
        mode,
        "performance",
        full_run=True,
        configuration=B128,
        validation_profile="performance-exploration",
        target_diagnostics="lean",
        target_dispatch=policy,
    )
    r = decode_scan_results.summarize(h.path, h.directory, h.point, probe=False)
    assert r["valid"], r.get("primary_error")
    assert h.runtime["diagnostic_configuration"]["target_dispatch"] == policy
    rows = [
        r
        for r in t.report()["rows"]
        if r["category"] == "control_snapshot_publish" and r["file_name"] == "s2-control.json"
    ]
    assert rows and all(r["encoding_policy"] == policy for r in rows)
    # Production constructor still carries original deadline, live population,
    # current claims and no-bonus K3; no optimized snapshot cache exists.
    packet = s2_pool.control()
    assert packet["active_limit"] == 128
    archive = tmp_path / "one.tar.gz"
    export(h.directory, archive, first_code=0, modes=MODES)
    with tarfile.open(archive) as tfile:
        paths = json.load(tfile.extractfile("inventory.json"))["logical_paths"]
        restored = json.load(tfile.extractfile(paths["runtime.json"]))
    assert restored["diagnostic_configuration"] == h.runtime["diagnostic_configuration"]
    assert restored["output_equivalence_status"] == "NOT_RUN"


@pytest.mark.parametrize("policy", ["reference", "encode-once"])
def test_production_alias_byte_equivalence_current_contents_and_write_count(
    tmp_path, monkeypatch, policy
):
    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", policy)
    original_open, original_replace = Path.open, Path.replace
    counts, replacements = [], []

    class Writer:
        def __init__(self, h):
            self.h = h
            self.calls = 0

        def __enter__(self):
            return self

        def write(self, text):
            self.calls += 1
            return self.h.write(text)

        def __exit__(self, *args):
            counts.append(self.calls)
            return self.h.__exit__(*args)

    def opening(p, *args, **kwargs):
        h = original_open(p, *args, **kwargs)
        return Writer(h) if p == path.with_suffix(".json.tmp") and args == ("w",) else h

    def replacing(p, target):
        replacements.append((str(p), str(target)))
        return original_replace(p, target)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(Path, "replace", replacing)
    import os

    monkeypatch.setattr(os, "fsync", lambda *_: pytest.fail("new control fsync"))
    for version in range(3):
        value = dict(
            deadline_ns=123456789,
            requests={"x": {"version": version, "state": "ACTIVE"}},
            proposal=[1, 2, 3],
            unicode="边界",
            flag=False,
        )
        expected = io.StringIO()
        json.dump(value, expected, allow_nan=False)
        fixed_runtime.publish(path, value)  # Real pre-existing imported production alias.
        assert path.read_text() == expected.getvalue()
        assert s2_pool.control() == value
    assert len(replacements) == 3
    assert counts == [1, 1, 1] if policy == "encode-once" else min(counts) > 1
    # Optimized policy does not touch a different publisher or state file.
    other = tmp_path / "drain-state.json"
    fixed_runtime.publish(other, {"deadline_ns": 123456789})
    assert json.loads(other.read_text()) == {"deadline_ns": 123456789}


@pytest.mark.parametrize("policy", ["reference", "encode-once"])
@pytest.mark.parametrize("fault", ["encoding", "write", "replace"])
def test_failure_never_publishes_success_or_changes_old_deadline(
    tmp_path, monkeypatch, policy, fault
):
    path = tmp_path / "s2-control.json"
    monkeypatch.setenv("SR_S2_CONTROL", str(path))
    monkeypatch.setenv("SR_K3_TARGET_DISPATCH", policy)
    value = {"deadline_ns": 987654321, "version": 1}
    s2_pool.publish(path, value)
    original = Path.open

    def broken(p, *args, **kwargs):
        if p == path.with_suffix(".json.tmp") and args == ("w",):
            raise OSError("injected write failure")
        return original(p, *args, **kwargs)

    if fault == "write":
        monkeypatch.setattr(Path, "open", broken)
    elif fault == "replace":

        def replace(*_):
            raise OSError("injected replace failure")

        monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises((ValueError, OSError)):
        s2_pool.publish(path, {**value, "version": float("nan") if fault == "encoding" else 2})
    assert s2_pool.control() == value


def test_profiles_do_not_change_policy_or_hide_missing_evidence():
    a, b = [declaration(x, B128) for x in ("lean-reference", "lean-dispatch-opt")]
    assert a.pop("configuration") != b.pop("configuration")
    assert a.pop("target_dispatch") == "reference"
    assert b.pop("target_dispatch") == "encode-once"
    assert a == b and a["target_diagnostics"] == "lean"
    assert qualification_errors({}, {}, {"target_dispatch": "encode-once"})
    assert qualification_errors({}, {}, {}) == []  # Historical contract stays historical.


@pytest.mark.parametrize('policy', ['reference', 'encode-once'])
def test_collected_control_cycle_qualification_then_archive_replay(policy, monkeypatch, tmp_path):
    import copy
    import time
    from test_k3_cycle_accounting import collected_clock_fixture

    from specrhythm.continuation import trace
    from specrhythm.serving.k3_cycle_evidence import compact_report, cycle_report
    from specrhythm.serving.k3_scale_report import metadata
    from specrhythm.serving.fixed_plan import settings

    r, b = collected_clock_fixture(monkeypatch)
    opts = settings(target_dispatch=policy, target_diagnostics='lean')
    r['diagnostic_configuration'] = metadata(opts)
    t = CausalTrace(enabled=True, layout='phased')
    monkeypatch.setattr(trace, 'TRACE', t)
    path = tmp_path/'s2-control.json'
    monkeypatch.setenv('SR_S2_CONTROL', str(path))
    monkeypatch.setenv('SR_K3_TARGET_DISPATCH', policy)
    original_clock = time.monotonic_ns
    for s in r['target_steps']:
        # GPU/transport endpoints in this CPU fixture use a controlled clock.
        ticks = iter(range(s['start_ns']-9, s['start_ns']+20))
        monkeypatch.setattr(time, 'monotonic_ns', lambda: next(ticks))
        fixed_runtime.publish(path, {'pp_admission': s['ping_admission'], 'deadline_ns': 99999})
    monkeypatch.setattr(time, 'monotonic_ns', original_clock)
    r['host'] = {'intervals': [], 'causal_timeline': t.report()}
    v = dict(cycle_accounting=compact_report(cycle_report(r, b)))
    assert not qualification_errors(v, r, opts)
    for name, obj in [('runtime.json', r), ('draft-backend-report.json', b)]:
        (tmp_path/name).write_text(json.dumps(obj))
    archive = tmp_path/'cycles.tar.gz'
    export(tmp_path, archive, first_code=23, modes=MODES)
    with tarfile.open(archive) as tar:
        inv = json.load(tar.extractfile('inventory.json'))
        assert inv['first_exit_code'] == 23
        restored = [json.load(tar.extractfile(inv['logical_paths'][name]))
                    for name in ('runtime.json', 'draft-backend-report.json')]
    rr, bb = restored
    assert not qualification_errors(dict(cycle_accounting=compact_report(cycle_report(rr, bb))),
                                    rr, opts)
    bad = copy.deepcopy(rr)
    bad['host']['causal_timeline']['rows'] = []
    assert qualification_errors(v, bad, opts)
