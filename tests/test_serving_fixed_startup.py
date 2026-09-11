"""Observed TP startup still invokes real S2 UUID binding and inherited verify hooks."""

from types import SimpleNamespace

import pytest
from test_serving_s2_uuid import hardware as _hardware
from test_serving_s2_uuid import phase4_config as _config
from test_serving_s2_uuid import startup as _startup

from specrhythm.serving import fixed_observe, s2_runtime

hardware = _hardware
phase4_config = _config
startup = _startup


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("matching", ["linear", "bound-prefix"])
def test_observed_startup_initializes_each_rank_once_before_verification(
    startup, monkeypatch, probe, matching
):
    h = startup
    from specrhythm.serving import fixed_logging

    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    monkeypatch.setenv("SR_FIXED_POINT", str(h.directory / "cpu-point.json"))
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)
    monkeypatch.setattr(fixed_observe, "install_host_observation", lambda: None)
    observed = []

    def timeline(torch, model, metadata, *, identity):
        observed.append(identity)
        return SimpleNamespace(report=lambda: {"identity": identity, "forwards": []})

    monkeypatch.setattr(fixed_observe, "DeviceTimeline", timeline)
    engine = s2_runtime.make_engine

    def construct(*args, **kwargs):
        llm = engine(*args, **kwargs)
        for worker in h.workers:
            worker.vllm_config.model_config.dtype = "bfloat16"
        rpc = llm.collective_rpc

        def observed_rpc(callback):
            return rpc(
                fixed_observe.target_startup
                if callback is s2_runtime.initialize_pingpong_worker
                else callback
            )

        llm.collective_rpc = observed_rpc
        return llm

    monkeypatch.setattr(s2_runtime, "make_engine", construct)
    result = h.run("pingpong", probe=probe)
    assert len(observed) == 2
    assert {r["physical_gpu_id"] for r in observed} == {1, 2}
    for worker in h.workers:
        from specrhythm.serving.fixed_identity import report

        assert report(worker.model_runner.drafter.identity)["mode"] == matching
        evidence = worker.model_runner.drafter.uuid_queries.evidence()
        assert evidence["uuid_initial_validation_count"] == 1
        assert evidence["uuid_query_mode"] == "live"
        assert evidence["uuid_verification_subprocess_query_count"] == (0 if probe else 1)
    assert h.shutdown and result["draft_shutdown"]["shutdown"]
