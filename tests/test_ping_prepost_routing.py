"""New mode configuration and actual production service construction; no CUDA loaded."""

import sys
from types import SimpleNamespace as NS

import pytest
from test_phase4_serial import phase4_config as _config
from test_serial_eager_owner import OwnerWorker

from specrhythm.continuation.ping_prepost_backend import PingPrePostBackendMixin
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend
from specrhythm.serving import eager_draft, fixed_runtime, s2_runtime
from specrhythm.serving.decode_scan_plan import selected_point
from specrhythm.serving.fixed_plan import capacity_metadata
from specrhythm.serving.ping_prepost import MODES
from specrhythm.serving.ping_prepost_machine import PingPrePostMachine
from specrhythm.serving.ping_prepost_owner import PingPrePostOwner
from specrhythm.serving.s2_pool import prefix_record

phase4_config = _config


@pytest.mark.parametrize("mode", MODES)
def test_true_service_factory_and_target_engine_classes(
    mode, phase4_config, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=lambda **kw: calls.append(kw)))
    s2_runtime.make_engine(
        phase4_config, mode, classes=fixed_runtime.CLASSES, sequence_limit=512, query_limit=4096
    )
    cfg = calls[0]
    assert cfg["scheduler_cls"].endswith("PingPrePostScheduler")
    assert cfg["speculative_config"]["model"].endswith("PingPrePostProposer")
    assert cfg["speculative_config"]["num_speculative_tokens"] == 4
    assert cfg["tensor_parallel_size"] == 2 and not cfg["async_scheduling"]
    cap = capacity_metadata(mode, active_limit=16, resident_requirement=360)
    assert cap["active_request_limit"] == 16 and cap["max_requests_per_target_forward"] == 8
    assert cap["per_cohort_capacity"] == 8 and cap["resident_request_requirement"] == 360
    with pytest.raises(ValueError, match="B16"):
        selected_point(mode, 32)
    owners = []

    class Hardware(VllmBatchedDraftBackend):
        def __init__(self, config):
            super().__init__(config, worker=OwnerWorker())

        def physical_rows(self):
            return {
                rid: prefix_record(s.prefix, s.materialized, [[id(s)]])
                for rid, s in self.states.items()
            }

    def server_serve(server, ready_path):
        owners.append(server.machine)
        assert isinstance(server.machine, PingPrePostOwner)
        assert isinstance(server.machine.machine, PingPrePostMachine)
        assert isinstance(server.machine.machine.backend, PingPrePostBackendMixin)
        assert server.machine.machine.enabled == (mode == MODES[1])
        assert server.machine.commands.maxsize == 256
        # No admitted work: exercise actual shutdown, owner join and final report.
        server.machine.call("shutdown", {})

    monkeypatch.setattr(eager_draft, "control", lambda: {"requests": {"r": {}}})
    monkeypatch.setattr(eager_draft.EagerSerialServer, "serve", server_serve)
    # Tests explicitly leave the pre-existing global logging layer unselected.
    from specrhythm.serving import fixed_logging

    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    eager_draft.serve(
        NS(max_model_len=4096),
        tmp_path,
        tmp_path / "unused.sock",
        backend_class=Hardware,
        prepost_mode=mode,
    )
    assert len(owners) == 1 and not owners[0]._thread.is_alive()
    assert (tmp_path / "draft-backend-report.json").is_file()


def test_initial_refill_adapter_uses_one_owner_registration(tmp_path, monkeypatch):
    from test_ping_prepost import Backend, cleanup
    from test_serial_eager_owner import proposal_row

    from specrhythm.phase4.serial import token_prefix_hash

    m = PingPrePostMachine(
        Backend(NS(max_model_len=4096), worker=OwnerWorker()), request_ids=["a"]
    )
    m.initialize("a", (10, 20), token_prefix_hash((10, 20)))
    clock = NS(rows={"a": {"cohort": "B"}}, definitions={"a": NS(maximum_new_tokens=101)})
    warm = {"a": NS(logical_committed_prefix_token_ids=(10, 20))}
    operations = []

    def call(operation, payload):
        operations.append(operation)
        assert operation == "pp_register"
        return m.register(payload["requests"])

    try:
        s2_runtime.initial_work(MODES[1], ["a"], clock, warm, NS(call=call), {"eos_token_ids": []})
        assert operations == ["pp_register"] and m.homes == {"a": "B"}
        assert len(m.ready["a"]["proposal"]["proposal_token_ids"]) == 1
        assert m.backend.metrics.forwards["proposal"] == 0
        with pytest.raises(ValueError, match="duplicate/moved"):
            m.register([{**proposal_row("a"), "home_cohort": "A"}])
    finally:
        cleanup(m)
