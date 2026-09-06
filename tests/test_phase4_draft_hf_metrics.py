from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from specrhythm.phase4 import draft_service
from specrhythm.phase4.draft_hf_metrics import MeasuredHFDraftBackend, MeasuredHFStateMachine
from specrhythm.phase4.manifest import sha256_file


def test_hf_measurement_delegates_tokens_and_collects_events_only_at_shutdown(
    monkeypatch, tmp_path
):
    calls = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            calls.append("event")

        def elapsed_time(self, end):
            assert "synchronize" in calls
            return 0.5

    class Model:
        def register_forward_pre_hook(self, hook, *, with_kwargs):
            assert with_kwargs
            self.before = hook
            return SimpleNamespace(remove=lambda: calls.append("remove-before"))

        def register_forward_hook(self, hook):
            self.after = hook
            return SimpleNamespace(remove=lambda: calls.append("remove-after"))

        def forward(self, size):
            self.before(self, (), {"input_ids": SimpleNamespace(numel=lambda: size)})
            calls.append(("forward", size))
            self.after(self, (), None)

    def initialize_backend(self, config):
        self.model = Model()
        self.states = {}
        self._provenance = {"model": "unchanged", "backend": self.backend_name}
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(Event=Event, synchronize=lambda: calls.append("synchronize"))
        )

    def initialize(self, rid, prefix):
        self.model.forward(len(prefix))
        self.states[rid] = SimpleNamespace(proposal=())

    def propose(self, rid, budget, eos):
        self.model.forward(1)
        self.states[rid].proposal = (9, 10)
        return (9, 10), 1

    monkeypatch.setattr(draft_service.HFPersistentDraftBackend, "__init__", initialize_backend)
    monkeypatch.setattr(draft_service.HFPersistentDraftBackend, "initialize", initialize)
    monkeypatch.setattr(draft_service.HFPersistentDraftBackend, "propose", propose)
    monkeypatch.setattr(
        draft_service.HFPersistentDraftBackend,
        "rollback",
        lambda self, rid, n: calls.append(("rollback", n)),
    )
    monkeypatch.setattr(
        draft_service.HFPersistentDraftBackend,
        "append_target_token",
        lambda self, rid, token: self.model.forward(1),
    )
    monkeypatch.setattr(
        draft_service.HFPersistentDraftBackend,
        "shutdown",
        lambda self: (calls.append("shutdown"), self.states.clear()),
    )
    backend = MeasuredHFDraftBackend(None)
    backend.initialize("r", [1, 2, 3])
    assert backend.propose("r", 2, []) == ((9, 10), 1)
    backend.rollback("r", 1)
    backend.append_target_token("r", 11)
    assert "synchronize" not in calls
    path = tmp_path / "draft-backend-report.json"
    machine = MeasuredHFStateMachine(backend, 4, path)
    result = machine.shutdown()
    evidence = json.loads(path.read_text())
    assert result["draft_backend_report_sha256"] == sha256_file(path)
    assert evidence["draft_model_forward_count"] == 2
    assert evidence["draft_model_forward_count_by_purpose"] == {
        "setup": 1,
        "proposal": 1,
        "commit": 1,
    }
    assert evidence["draft_batch_size_p50"] == 1
    assert evidence["draft_gpu_event_time_ms"] == 1.0
    assert evidence["draft_proposed_token_count"] == 2
    assert evidence["draft_kv_operations"]["invalidated_tokens"] == 1
    assert evidence["draft_scalar_item_count"] == 2
    assert calls.count("synchronize") == 1 and calls.count("shutdown") == 1
    backend.shutdown()
    assert calls.count("synchronize") == 1 and calls.count("shutdown") == 1


def test_default_hf_service_and_dual_do_not_install_measurement(monkeypatch, tmp_path):
    monkeypatch.delenv("SR_PHASE4B3_HF_METRICS", raising=False)
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "hf-persistent")
    constructed = []
    monkeypatch.setattr(
        draft_service,
        "HFPersistentDraftBackend",
        lambda config: constructed.append("ordinary-hf") or object(),
    )
    monkeypatch.setattr(
        draft_service, "DraftUnixServer", lambda *a, **k: SimpleNamespace(serve=lambda path: None)
    )
    draft_service.run_draft_service(
        SimpleNamespace(proposal_budget=4),
        socket_path=tmp_path / "s",
        event_log_path=tmp_path / "e",
        ready_path=tmp_path / "r",
    )
    assert constructed == ["ordinary-hf"]
    root = Path("src/specrhythm/phase4")
    for path in (*root.glob("dual*.py"), *root.glob("vllm_dual*.py")):
        assert "draft_hf_metrics" not in path.read_text()
        assert "SR_PHASE4B3_HF_METRICS" not in path.read_text()


def test_serial_metrics_selector_dispatches_only_to_explicit_observer(monkeypatch, tmp_path):
    from specrhythm.phase4 import draft_hf_metrics

    called = []
    monkeypatch.setenv("SR_PHASE4B3_HF_METRICS", "1")
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "hf-persistent")
    monkeypatch.setattr(draft_hf_metrics, "serve_measured_hf", lambda *a, **kw: called.append(kw))
    draft_service.run_draft_service(
        None, socket_path=tmp_path / "s", event_log_path=tmp_path / "e", ready_path=tmp_path / "r"
    )
    assert len(called) == 1
