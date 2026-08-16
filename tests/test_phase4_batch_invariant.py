import ast
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrhythm.phase4.batch_invariant import (
    configure_before_worker_creation,
    pinned_vllm_hardware_supported,
    probe_batch_invariant_hardware,
    require_matching_reference_mode,
    validate_batch_invariant_ranks,
)
from specrhythm.phase4.fixed_control import run_fixed_proposal_service
from specrhythm.phase4.reference import _exclusive_freeze
from specrhythm.phase4.serial_runner import PATCHED_VLLM_RUNNER_SHA256
from specrhythm.phase4.transport import UnixDraftClient
from specrhythm.phase4.vllm_diagnostics import (
    compare_divergence_diagnostics,
    compare_fixed_proposal_controls,
    logits_position_mapping,
    token_sha256,
    validate_kv_monotonicity,
    validate_logits_mapping,
    validate_target_diagnostic,
)
from specrhythm.phase4.vllm_remote import _assert_target_information_isolated


def rank_evidence(rank=0, *, requested=True, capability="8.0", effective=True):
    return {
        "global_rank": rank,
        "batch_invariant_env_raw": "1" if requested else "0",
        "batch_invariant_requested": requested,
        "batch_invariant_env_resolved": requested,
        "batch_invariant_effective": effective,
        "batch_invariant_validation": {"valid": effective, "reasons": []},
        "compute_capability": capability,
        "documented_hardware_supported": pinned_vllm_hardware_supported(capability),
        "disable_custom_all_reduce": requested,
        "all_reduce_backends": ["PYNCCL"],
        "attention_batch_invariance": [
            {"backend": "FLASH_ATTN", "supports_batch_invariance": True}
        ],
        "cascade_attention_enabled": False,
        "vllm_dbo_enabled": False,
        "dtype": "torch.bfloat16",
    }


def diagnostic(*, proposal=(10, 11), prefix=(1, 2), kv_length=2):
    mapping = logits_position_mapping(
        proposal, sampled_logits_offset=0, flattened_input_offset=0
    )
    return {
        "schema_version": "specrhythm.phase4-target-forward-diagnostic.v1",
        "request_id": "request",
        "committed_prefix_token_ids": list(prefix),
        "committed_prefix_sha256": token_sha256(prefix),
        "proposal_token_ids": list(proposal),
        "logical_target_kv_length": kv_length,
        "scheduled_token_count": len(proposal) + 1,
        "query_length": len(proposal) + 1,
        "sequence_length": kv_length + len(proposal) + 1,
        "logits_position_mapping": mapping,
        "position_ids": list(range(kv_length, kv_length + len(proposal) + 1)),
        "attention_mask_proof": {
            "causal": True,
            "query_start_range": [0, len(proposal) + 1],
            "query_length_matches_scheduler": True,
            "attention_sequence_length": kv_length + len(proposal) + 1,
            "positions_contiguous": True,
        },
        "top_raw_logits": [[{"token_id": 10, "raw_logit": 1.0}]],
        "top_target_logprobs": [[{"token_id": 10, "log_probability": -0.1}]],
        "selected_target_token_id": [10],
        "target_verification_shape": {
            "request_count": 1,
            "scheduled_input_positions": len(proposal) + 1,
            "sampled_logits_rows": len(proposal) + 1,
            "vocab_size": 100,
        },
        "attention_backend": ["FLASH_ATTN"],
        "all_reduce_backend": "PYNCCL",
        "dtype": "torch.bfloat16",
        "batch_invariant_requested": True,
        "causal_attention": True,
        "target_kv_contains_rejected_or_future_tokens": False,
        "visible_to_draft": False,
    }


def test_batch_invariant_env_is_set_before_vllm_import():
    environment = {}
    evidence = configure_before_worker_creation(
        "batch-invariant", environ=environment, loaded_modules=[]
    )
    assert environment["VLLM_BATCH_INVARIANT"] == "1"
    assert evidence["configured_before_vllm_import"] is True
    with pytest.raises(RuntimeError, match="before importing vLLM"):
        configure_before_worker_creation(
            "batch-invariant", environ={}, loaded_modules=["vllm.envs"]
        )


def test_batch_invariant_preflight_is_explicit_without_cuda(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    report = probe_batch_invariant_hardware("batch-invariant")
    assert report["valid"] is False
    assert report["batch_invariant_effective"] is False
    assert report["errors"] == ["CUDA is unavailable"]


@pytest.mark.parametrize(
    ("capability", "supported"),
    [("7.5", False), ("8.0", True), ("9.0", True)],
)
def test_pinned_vllm_compute_capability_boundary(capability, supported):
    assert pinned_vllm_hardware_supported(capability) is supported


def test_a800_preflight_proceeds_but_requires_initialized_worker_evidence(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_capability=lambda _index: (8, 0),
            get_device_name=lambda _index: "NVIDIA A800-SXM4-80GB",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    report = probe_batch_invariant_hardware("batch-invariant")
    assert report["valid"] is True
    assert report["errors"] == []
    assert all(row["documented_hardware_supported"] for row in report["devices"])
    assert report["batch_invariant_effective"] is False
    assert report["effective_requires_initialized_worker_evidence"] is True


def test_pre_ampere_preflight_fails_at_the_correct_pinned_boundary(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_capability=lambda _index: (7, 5),
            get_device_name=lambda _index: "Pre-Ampere test device",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    report = probe_batch_invariant_hardware("batch-invariant")
    assert report["valid"] is False
    assert report["devices"][0]["documented_hardware_supported"] is False
    assert any("compute capability >= 8.0" in error for error in report["errors"])
    assert report["batch_invariant_effective"] is False


def test_target_only_and_serial_reference_mode_must_match():
    reference = {"target_runtime_configuration": {"correctness_mode": "default"}}
    require_matching_reference_mode(reference, "default")
    with pytest.raises(ValueError, match="does not match"):
        require_matching_reference_mode(reference, "batch-invariant")


def test_requested_effective_validation_fails_closed_and_checks_ranks():
    rows = [rank_evidence(0), rank_evidence(1)]
    assert validate_batch_invariant_ranks(rows, requested=True)[
        "batch_invariant_effective"
    ]
    rows[1] = rank_evidence(1, capability="7.5", effective=False)
    rows[1]["batch_invariant_validation"]["reasons"] = ["unsupported hardware"]
    report = validate_batch_invariant_ranks(rows, requested=True)
    assert not report["batch_invariant_effective"]
    assert any("cannot prove" in error for error in report["batch_invariant_validation"]["errors"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_invariant_env_resolved", False, "did not resolve"),
        ("documented_hardware_supported", False, "hardware contract"),
        ("disable_custom_all_reduce", False, "custom all-reduce"),
        ("cascade_attention_enabled", True, "cascade attention"),
        ("vllm_dbo_enabled", True, "DBO"),
    ],
)
def test_requested_batch_invariance_requires_every_worker_condition(
    field, value, message
):
    row = rank_evidence()
    row[field] = value
    report = validate_batch_invariant_ranks([row], requested=True)
    assert report["batch_invariant_effective"] is False
    assert any(
        message in error for error in report["batch_invariant_validation"]["errors"]
    )


def test_requested_batch_invariance_requires_attention_backend_evidence():
    row = rank_evidence()
    row["attention_batch_invariance"] = [
        {"backend": "OTHER", "supports_batch_invariance": False}
    ]
    report = validate_batch_invariant_ranks([row], requested=True)
    assert report["batch_invariant_effective"] is False
    assert any(
        "every active attention backend" in error
        for error in report["batch_invariant_validation"]["errors"]
    )


def test_divergent_position_proposal_logits_mapping():
    row = diagnostic()
    assert not validate_logits_mapping(row)
    row["logits_position_mapping"][1]["proposal_index"] = 0
    assert "proposal index is not contiguous" in validate_logits_mapping(row)


def test_target_diagnostic_proves_prefix_positions_and_kv():
    row = diagnostic()
    assert not validate_target_diagnostic(row)
    proof = compare_divergence_diagnostics(
        [row],
        [json.loads(json.dumps(row))],
        request_id="request",
        committed_prefix_sha256=row["committed_prefix_sha256"],
    )
    assert proof["valid"], proof


def test_rejected_kv_rollback_and_monotonicity():
    valid = [
        {
            "request_id": "r",
            "parent_prefix_len": 10,
            "committed_tokens": 2,
            "logical_target_kv_length": 12,
            "terminal": False,
        },
        {
            "request_id": "r",
            "parent_prefix_len": 12,
            "committed_tokens": 1,
            "logical_target_kv_length": 13,
            "terminal": True,
        },
    ]
    assert not validate_kv_monotonicity(valid)
    contaminated = json.loads(json.dumps(valid))
    contaminated[0]["logical_target_kv_length"] = 14
    assert any("commit accounting" in error for error in validate_kv_monotonicity(contaminated))


def test_local_remote_fixed_proposal_equivalence():
    evidence = {
        "proposal_token_ids": [53143, 2213, 369, 264],
        "top_raw_logits": [[{"token_id": 53143, "raw_logit": 1.0}]],
        "top_target_logprobs": [[{"token_id": 53143, "log_probability": -0.2}]],
        "accepted_prefix_length": 1,
        "committed_token_ids": [53143, 99],
    }
    assert compare_fixed_proposal_controls(evidence, dict(evidence))[
        "local_remote_equal"
    ]
    remote = dict(evidence)
    remote["accepted_prefix_length"] = 0
    assert not compare_fixed_proposal_controls(evidence, remote)["local_remote_equal"]


def test_remote_fixed_service_preserves_proposal_bytes():
    with tempfile.TemporaryDirectory(prefix="sr-phase4-", dir="/tmp") as directory:
        socket_path = Path(directory) / "fixed.sock"
        thread = threading.Thread(
            target=run_fixed_proposal_service, args=(socket_path,), daemon=True
        )
        thread.start()
        for _ in range(100):
            if socket_path.is_socket():
                break
            time.sleep(0.01)
        assert socket_path.is_socket()
        client = UnixDraftClient(socket_path)
        rows = [[53143, 2213, 369, 264]]
        assert client.call("fixed_proposal", {"rows": rows})["rows"] == rows
        assert client.shutdown()["shutdown"] is True
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not socket_path.exists()


def test_target_diagnostics_are_not_draft_visible():
    _assert_target_information_isolated({"proposal_token_ids": [1, 2]})
    with pytest.raises(ValueError, match="leaked"):
        _assert_target_information_isolated({"top_raw_logits": [[1.0]]})


def test_reference_freeze_requires_fresh_artifact(tmp_path):
    path = tmp_path / "fresh" / "reference.json"
    _exclusive_freeze(path, {"immutable": True})
    with pytest.raises(FileExistsError):
        _exclusive_freeze(path, {"immutable": True})


def test_python311_and_pinned_patch_provenance_are_parseable():
    root = Path(__file__).parents[1]
    config = json.loads(
        (root / "configs" / "phase4a_target_fair_1d2v.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["framework_freeze"]["python_series"] == "3.11"
    for name in (
        "batch_invariant.py",
        "correctness_validation.py",
        "fixed_control.py",
        "vllm_diagnostics.py",
    ):
        path = root / "src" / "specrhythm" / "phase4" / name
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 11))
    manager = (root / "integrations" / "vllm" / "manage_patch.py").read_text(
        encoding="utf-8"
    )
    assert "752a3a504485790a2e8491cacbb35c137339ad34" in manager
    assert PATCHED_VLLM_RUNNER_SHA256 in manager
