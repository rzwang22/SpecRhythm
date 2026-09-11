"""Actual fixed scheduler -> resident/Dual gates -> identity binding, CPU stock allocator."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_dual_scheduler import DraftClientStub, Request, proposal_result
from test_serving_s2 import s2_schedulers as _schedulers

from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.serving.common import DataError
from specrhythm.serving.fixed_identity import BoundPromptIdentityMap, install, qualify, report
from specrhythm.serving.fixed_plan import settings
from specrhythm.serving.s2_pool import publish

s2_schedulers = _schedulers


def test_actual_full_scans_reduce_and_live_rows_are_always_read(monkeypatch):
    prompts = {str(i): (1, i, 7) for i in range(100)}
    scans = []
    original = FrozenPromptIdentityMap.match

    def scanned(self, tokens):
        scans.append(id(self))
        return original(self, tokens)

    monkeypatch.setattr(FrozenPromptIdentityMap, "match", scanned)
    linear = FrozenPromptIdentityMap(prompts)
    # A/B are independent owners; installation within one owner now preserves aliases.
    fast = BoundPromptIdentityMap(FrozenPromptIdentityMap(prompts))
    # Full initial binding, then actual changing output prefixes / versions.
    for round_id in range(4):
        for rid, prompt in prompts.items():
            current = list(prompt) + [round_id] * (round_id + 1)
            assert linear.bind("opaque-" + rid, current) == fast.bind("opaque-" + rid, current)
    assert scans.count(id(linear)) == 400
    assert scans.count(id(fast)) == 100
    assert fast.report()["candidate_comparisons"] == 10_300
    assert fast.report()["linear_candidate_comparisons"] == 40_000
    assert linear.internal_to_stable == fast.internal_to_stable
    assert linear.stable_to_internal == fast.stable_to_internal


@pytest.mark.parametrize("prompts", [{"a": [1], "b": [1, 2]}, {"a": [1, 2], "b": [3]}])
def test_changed_refilled_cancelled_and_invalid_rows_keep_exact_failures(prompts):
    owners = [FrozenPromptIdentityMap(prompts)]
    owners.append(BoundPromptIdentityMap(FrozenPromptIdentityMap(prompts)))
    # Includes short, missing, ambiguous, changed identity, reused released internal
    # ID and duplicate stable alias. Historical binding lifetime is intentionally retained.
    operations = [
        ("", []), ("x", [1]), ("x", [1, 2]), ("x", [1, 2, 9]),
        ("x", [3]), ("y", [3]), ("y", [1, 2]), ("z", [99]),
        ("x", ["bad"]), ("x", []), ("x", [1, 2, 7]),
    ]
    for internal, current in operations:
        outcomes = []
        for owner in owners:
            try:
                outcomes.append(("ok", owner.bind(internal, current)))
            except (ValueError, RuntimeError) as e:
                outcomes.append((type(e).__name__, str(e)))
        assert outcomes[0] == outcomes[1]
        assert owners[0].internal_to_stable == owners[1].internal_to_stable
        assert owners[0].stable_to_internal == owners[1].stable_to_internal
    if not owners[1].prefix_free:
        assert owners[1].report()["validated_binding_reuses"] == 0


def test_frozen_proof_cannot_be_stale_and_owners_are_isolated():
    source = {"a": [1, 2]}
    first = BoundPromptIdentityMap(FrozenPromptIdentityMap(source))
    second = BoundPromptIdentityMap(FrozenPromptIdentityMap({"b": [3, 4]}))
    source["a"][:] = [99]
    assert first.bind("same-internal", [1, 2, 9]) == "a"
    assert second.bind("same-internal", [3, 4, 9]) == "b"
    with pytest.raises(TypeError):
        first.stable_prompts["a"] = (9,)
    first.stable_prompts = {"a": (9,)}
    with pytest.raises(DataError, match="table was replaced"):
        first.bind("same-internal", [9])


def test_threaded_owner_binding_serializes_without_alias_or_lost_counters():
    owner = BoundPromptIdentityMap(FrozenPromptIdentityMap({"a": [1, 2]}))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda i: owner.bind("x", [1, 2, i]), range(100))) == ["a"] * 100
    assert owner.report()["full_scans"] == 1
    assert owner.report()["validated_binding_reuses"] == 99


def test_invalid_generated_suffix_still_normalizes_and_counts_no_candidate_search():
    owner = BoundPromptIdentityMap(FrozenPromptIdentityMap({"a": [1, 2]}))
    owner.bind("x", [1, 2])
    before = owner.report()
    with pytest.raises(ValueError):
        owner.bind("x", [1, 2, "bad"])
    after = owner.report()
    assert after["candidate_comparisons"] == before["candidate_comparisons"]
    assert after["linear_candidate_comparisons"] == before["linear_candidate_comparisons"]
    assert after["binding_errors"] == 1
    assert owner.bind("x", [1, 2, 7]) == "a"


def test_explicit_configuration_default_and_idempotent_install(monkeypatch):
    assert settings()["identity_matching"] == "linear"
    assert settings(identity_matching="bound-prefix")["observation"] == "original-live"
    with pytest.raises(DataError):
        settings(identity_matching="guess")
    owner = SimpleNamespace(identity=FrozenPromptIdentityMap({"a": [1]}))
    original = owner.identity
    monkeypatch.delenv("SR_FIXED_IDENTITY_MATCHING", raising=False)
    install(owner, "identity")
    assert owner.identity is original
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", "bound-prefix")
    monkeypatch.delenv("SR_FIXED_POINT", raising=False)
    with pytest.raises(DataError):
        install(owner, "identity")
    monkeypatch.setenv("SR_FIXED_POINT", "point.json")
    install(owner, "identity")
    fast = owner.identity
    install(owner, "identity")
    assert owner.identity is fast
    assert type(FrozenPromptIdentityMap({"a": [1]})) is FrozenPromptIdentityMap  # S1/S2 default


def test_capacity_zero_access_and_selection_evidence_fail_closed():
    empty = report(BoundPromptIdentityMap(FrozenPromptIdentityMap({"a": [1]})))
    runtime = {"identity_matching": empty, "target_devices": [
        {"device": {"identity": {"global_rank": rank}}, "identity_matching": empty}
        for rank in (0, 1)
    ]}
    assert qualify(runtime, "bound-prefix")["by_owner"]["scheduler"]["bind_calls"] == 0
    with pytest.raises(DataError, match="selection differs"):
        qualify(runtime, "linear")
    runtime["target_devices"][0].pop("identity_matching")
    with pytest.raises(DataError, match="selection differs"):
        qualify(runtime, "bound-prefix")


@pytest.fixture
def fixed_schedulers(s2_schedulers, monkeypatch, tmp_path):
    _, path = s2_schedulers
    spec = importlib.util.spec_from_file_location(
        "cpu_fixed_scheduler", Path(__file__).resolve().parents[1]
        / "src/specrhythm/serving/fixed_scheduler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definitions = [SimpleNamespace(request_id=str(i), prompt_token_ids=(1, i + 1))
                   for i in range(100)]
    for name in ("resident_scheduler", "vllm_dual_scheduler"):
        target = sys.modules["specrhythm.phase4." + name]
        monkeypatch.setattr(target, "load_smoke_requests", lambda *a, **kw: definitions)
    dual = sys.modules["specrhythm.phase4.vllm_dual_scheduler"]
    monkeypatch.setattr(dual, "DualDraftClient", lambda *a, **kw: DraftClientStub())
    for key, val in {
        "SR_FIXED_POINT": "cpu-point.json", "SR_PHASE4_RESIDENT_SETUP": "1",
        "SR_PHASE4_RESIDENT_SETUP_READY": str(tmp_path / "ready"),
        "SR_PHASE4_DECODE_READY_MANIFEST": str(tmp_path / "manifest"),
        "SR_PHASE4_RESIDENT_ADMISSION_EVENTS": str(tmp_path / "admission.jsonl"),
        "SR_PHASE4_RESIDENT_INITIAL_PROPOSAL_EVENTS": str(tmp_path / "initial.jsonl"),
        "SR_PHASE4_DUAL_BATCH": "1", "SR_PHASE4_DUAL_DRAFT_SOCKET": str(tmp_path / "d.sock"),
        "SR_PHASE4_DUAL_SCHEDULER_EVENTS": str(tmp_path / "events.jsonl"),
        "SR_PHASE4_WORKLOAD": str(tmp_path / "work"), "SR_PHASE4_REQUEST_COUNT": "100",
        "SR_PHASE4_DUAL_RESIDENT": "0", "SR_PHASE4_DUAL_MICROBATCH_SIZE": "32",
        "SR_PHASE4_DUAL_TEST_COORDINATION": "none",
    }.items():
        monkeypatch.setenv(key, val)

    def build(mode, matching, *, batch=64, n=100):
        definitions[:] = [SimpleNamespace(request_id=str(i), prompt_token_ids=(1, i + 1))
                          for i in range(n)]
        monkeypatch.setenv("SR_PHASE4_REQUEST_COUNT", str(n))
        monkeypatch.setenv("SR_PHASE4_DUAL_MICROBATCH_SIZE", str(batch // 2))
        grouped = mode in ("pingpong", "serial-split")
        monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)
        monkeypatch.setenv("SR_PHASE4_RESIDENT_CONSUMER", "target-only" if mode == "target"
                           else "serial")
        packet = {
            "barrier_ns": 100, "active_limit": batch,
            "max_requests_per_target_forward": batch // 2 if grouped else batch,
            "requests": {str(i): {"state": "ACTIVE" if i < batch else "STAGED",
                                   "cohort": ("A" if i < batch // 2 else "B") if grouped else None}
                         for i in range(n)},
        }
        publish(path, packet)
        cls = module.FixedPingScheduler if grouped else (
            module.FixedSerialScheduler if mode == "serial" else module.FixedTargetScheduler
        )
        scheduler = cls()
        scheduler.requests = {str(i): Request(str(i), (1, i + 1)) for i in range(n)}
        scheduler.running = list(scheduler.requests.values())
        for row in scheduler.requests.values():
            row.all_token_ids.append(20)
            row.num_output_tokens = 2
            row.num_computed_tokens = 3
            row.spec_token_ids = [] if mode == "target" or grouped else [11, 12, 13, 14]
        scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=lambda i: [[int(i) + 1]])
        if grouped:
            scheduler._bind_vllm_requests()
            for i in range(batch):
                row = scheduler.requests[str(i)]
                scheduler._accept_ready_result(proposal_result(
                    str(i), prefix=tuple(row.all_token_ids), tokens=(11, 12, 13, 14)
                ))
        else:
            scheduler._resident_ready = {"valid": True}
            scheduler._bind_requests()
        scheduler.freeze_pool()
        return scheduler, packet
    return build, path


@pytest.mark.parametrize("mode", ["target", "serial", "serial-split", "pingpong"])
def test_real_scheduler_selection_kv_checks_and_accounting_equal(fixed_schedulers, mode):
    build, path = fixed_schedulers
    outcomes = []
    for matching in ("linear", "bound-prefix"):
        s, packet = build(mode, matching)
        result = s.schedule()
        grouped = mode in ("serial-split", "pingpong")
        assert len(result.num_scheduled_tokens) == (32 if grouped else 64)
        outcomes.append((result.num_scheduled_tokens, result.scheduled_spec_decode_tokens,
                         s.s2_steps[-1]["rows"]))
        if matching == "bound-prefix":
            evidence = s.s2_steps[-1]["identity_matching"]
            assert evidence["validated_binding_reuses"] == 100
            assert evidence["full_scans"] == 0
            assert evidence["candidate_comparisons"] == 100  # old full scan: 100 * 100
            snapshot = {"identity_matching": report(s._identity()),
                        "target_steps": [{**s.s2_steps[-1], "window": True}]}
            assert qualify(snapshot, matching)["measured_scheduler"]["full_scans"] == 0
            assert qualify(snapshot, matching)["measured_scheduler"]["bind_calls"] == 100
        if grouped:
            # Stock CPU substitute supplies the real post-verification row update;
            # real ready-result handling must still validate a fresh prefix/version.
            row = s.requests["0"]
            row.all_token_ids.extend([11, 12])
            row.num_output_tokens += 2
            row.num_computed_tokens += 2
            row.spec_token_ids = []
            s._accept_ready_result(proposal_result(
                "0", prefix=tuple(row.all_token_ids), version=2, round_id=1,
                tokens=(21, 22, 23, 24)
            ))
            assert s._dual_proposals["0"].prefix_version == 2
            with pytest.raises(RuntimeError, match="prefix_version"):
                s._accept_ready_result(proposal_result(
                    "0", prefix=tuple(row.all_token_ids), version=99, round_id=2,
                    tokens=(31, 32, 33, 34)
                ))
        # A live resident KV change still fails through physical_rows -> PoolAudit.
        s.kv_cache_manager.get_block_ids = lambda i: [[999]]
        with pytest.raises(DataError, match="block"):
            s.schedule()
        s.kv_cache_manager.get_block_ids = lambda i: [[int(i) + 1]]
        # Cancellation/release removes the physical request. Refill admits a new
        # resident; identity binding is retained exactly as in the original path.
        del s.requests["0"]
        s.running = list(s.requests.values())
        packet["requests"]["0"].update(state="CANCELLED", resources_released=True)
        packet["requests"]["64"].update(state="ACTIVE", cohort="A" if grouped else None)
        publish(path, packet)
        if grouped:
            s.selected_cohort = "B"
        after = s.schedule()
        assert "0" not in after.num_scheduled_tokens
        assert s._identity().stable_id("0") == "0"
        # Mutating a bound prompt cannot be hidden by a reused result, even after failure.
        s.requests["1"].all_token_ids[0] = 9999
        with pytest.raises(RuntimeError, match="no frozen workload prompt"):
            s.schedule()
    assert outcomes[0] == outcomes[1]
