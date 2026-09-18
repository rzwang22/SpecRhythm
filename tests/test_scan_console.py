"""Actual drive -> summarize -> retained publication, bounded terminal output only."""

import json

import pytest
from test_prepost_runtime_kind import driven as _driven
from test_prepost_runtime_kind import hardware as _hardware
from test_prepost_runtime_kind import produced as _produced

from specrhythm.serving import decode_scan_results
from specrhythm.serving.k3 import B128, MODES

hardware, produced, driven = _hardware, _produced, _driven


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stage", ["capacity_probe", "performance"])
def test_k3_actual_result_publication_is_quiet_and_lossless(mode, stage, driven, capsys):
    h = driven(mode, stage, full_run=True, configuration=B128,
               validation_profile="performance-exploration", target_diagnostics="lean")
    r = decode_scan_results.summarize(h.path, h.directory, h.point,
                                      probe=stage == "capacity_probe")
    assert r["valid"], r.get("primary_error")
    capsys.readouterr()
    result = decode_scan_results.emit_result(h.directory, r, h.point)
    out = capsys.readouterr().out
    assert len(out.encode()) < 4096
    assert out.count("[decode scan]") == 1
    display = json.loads(out.removeprefix("[decode scan] "))
    assert display["mode"] == mode and display["effective_exit_code"] == 0
    assert "admission_order" not in display and "diagnostic_logging" not in display
    assert json.loads((h.directory / "light-summary.json").read_text()) == result
    assert json.loads((h.directory / "result.json").read_text()) == result
    assert display["cleanup_status"] == result["cleanup_status"] == "PASS"


def test_failed_result_write_does_not_print_success(driven, capsys, monkeypatch):
    h = driven("serial-k3", "capacity_probe", full_run=True, configuration=B128,
               validation_profile="performance-exploration", target_diagnostics="lean")
    r = decode_scan_results.summarize(h.path, h.directory, h.point, probe=True)
    capsys.readouterr()

    def failed(*args):
        raise OSError("injected result publication")

    monkeypatch.setattr(decode_scan_results, "write_once", failed)
    with pytest.raises(OSError, match="result publication"):
        decode_scan_results.emit_result(h.directory, r, h.point)
    assert "[decode scan]" not in capsys.readouterr().out
