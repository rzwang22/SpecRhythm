"""Observed TP startup still invokes real S2 UUID binding and inherited verify hooks."""

import time
from types import SimpleNamespace

import pytest
from test_serving_s2_uuid import hardware as _hardware
from test_serving_s2_uuid import phase4_config as _config
from test_serving_s2_uuid import startup as _startup

from specrhythm.serving import fixed_observe, s2_runtime
from specrhythm.serving.common import DataError, read_json
from specrhythm.serving.fixed_identity import install, report
from specrhythm.serving.fixed_plan import point

hardware = _hardware
phase4_config = _config
startup = _startup


@pytest.fixture
def observed_startup(request, monkeypatch):
    """Real S2 LLM/proposer constructors and the fixed worker install RPC."""
    import test_serving_s2_uuid as startup_contract
    from test_serving_s1 import request as definition

    from specrhythm.serving import fixed_logging

    profile = startup_contract.profile
    monkeypatch.setattr(startup_contract, "profile", lambda *a, **kw:
                        profile(*a, rows=[definition(i, 32) for i in range(4)], **kw))
    h = request.getfixturevalue("startup")
    h.original_maps = []
    h.prebind = False
    monkeypatch.setattr(fixed_logging, "_CURRENT", None)
    monkeypatch.setenv("SR_FIXED_POINT", str(h.directory / "cpu-point.json"))
    monkeypatch.setattr(fixed_observe, "install_host_observation", lambda: None)
    monkeypatch.setattr(fixed_observe, "DeviceTimeline", lambda *a, identity, **kw:
                        SimpleNamespace(report=lambda: {"identity": identity, "forwards": []}))
    make_engine = s2_runtime.make_engine

    def construct(*args, **kwargs):
        llm = make_engine(*args, **kwargs)
        for worker in h.workers:
            worker.vllm_config.model_config.dtype = "bfloat16"
            proposer = worker.model_runner.drafter
            h.original_maps.append(proposer.identity)
            if h.prebind and proposer.tp_rank == 0:
                rid, prompt = next(iter(proposer.identity.stable_prompts.items()))
                assert proposer.identity.bind("opaque-0", prompt) == rid
        rpc = llm.collective_rpc
        initialized = False

        def observed_rpc(callback):
            nonlocal initialized
            if not initialized:
                assert callback in (s2_runtime.target_snapshot,
                                    s2_runtime.initialize_pingpong_worker)
                initialized = True
                return rpc(fixed_observe.target_startup)
            return rpc(callback)

        llm.collective_rpc = observed_rpc
        return llm

    monkeypatch.setattr(s2_runtime, "make_engine", construct)
    return h


def serial_verifications(h, *, fault=None):
    """Substitute resident GPU outputs/RPC only; keep binding and all hooks/accounting."""
    from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal, token_prefix_hash
    from specrhythm.phase4.vllm_remote import _TargetRequest
    from specrhythm.serving.s2_pool import publish

    p = h.workers[0].model_runner.drafter
    definitions = list(p.definitions.values())[:2]
    ids = ["opaque-0", "opaque-1"]
    stable = [d.request_id for d in definitions]

    def proposal(rid, round_id, prefix_len, prefix_hash):
        return Proposal(PROTOCOL_VERSION, rid, round_id, prefix_len, prefix_hash,
                        (200, 201, 202, 203), False, time.monotonic_ns(),
                        time.monotonic_ns(), 1, {}, {})

    packet = {"requests": {}, "initial_proposals": {}}
    # Resident bootstrap outputs are the hardware substitute, not identity aliases.
    for internal, d in zip(ids, definitions):
        prefix = d.prompt_token_ids + (100,)
        assert p.identity.bind(internal, prefix) == d.request_id
        p.requests[d.request_id] = _TargetRequest(
            d.request_id, d.prompt_token_ids, d.maximum_new_tokens, prefix, (100,),
            bootstrap_target_tokens=1,
        )
        packet["requests"][d.request_id] = {"state": "ACTIVE", "cohort": None}
        packet["initial_proposals"][d.request_id] = {
            "proposal": proposal(
                d.request_id, 0, len(prefix), token_prefix_hash(prefix)).to_dict(),
            "service_send_ns": time.monotonic_ns(), "transport_end_ns": time.monotonic_ns(),
        }
    if fault == "proposal":
        packet["initial_proposals"][stable[0]]["proposal"]["parent_prefix_hash"] = "stale"
    publish(h.directory / "s2-control.json", packet)
    if fault == "changed":
        p.identity.bind(ids[0], definitions[1].prompt_token_ids)
    if fault == "alias":
        p.identity.bind("alias", definitions[0].prompt_token_ids)
    if fault == "unbound":
        ids[0] = "unbound"

    def rpc(operation, payload):
        assert operation == "synchronize_and_batch_propose"
        start, end = time.monotonic_ns(), time.monotonic_ns()
        return {
            "synchronizations": [
                {"request_id": row["request_id"],
                 "decision": {"committed_token_ids": row["committed_delta"]},
                 "state": {"logical_draft_kv_length":
                           len(p.requests[row["request_id"]].committed_token_ids)
                           + len(row["committed_delta"])}}
                for row in payload["synchronizations"]
            ],
            "proposals": [proposal(row["request_id"], row["round_id"],
                                   row["committed_prefix_len"],
                                   row["committed_prefix_hash"]).to_dict()
                          for row in payload["proposals"]],
            "state_sync_start_ns": start, "state_sync_end_ns": end,
            "service_send_ns": time.monotonic_ns(), "transport_end_ns": time.monotonic_ns(),
        }

    class TokenRows:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            row, selection = index
            return SimpleNamespace(tolist=lambda: self.rows[row][selection])

    p.client = SimpleNamespace(call=rpc)
    h.verify_times = []
    for round_id in range(3):
        scheduled = dict.fromkeys(ids, [200, 201, 202, 203])
        for rank in (0, 1):
            with h.on_rank(rank):
                h.workers[rank].model_runner.drafter.on_target_verify_start(
                    request_ids=ids, scheduled_spec_token_ids=scheduled)
        if fault == "duplicate":
            p.on_target_verify_start(request_ids=ids, scheduled_spec_token_ids=scheduled)
        # Model output: two accepted candidates and one Target correction, no EOS.
        for rank in (0, 1):
            with h.on_rank(rank):
                h.workers[rank].model_runner.drafter.on_target_verify_end(
                    request_ids=ids, sampled_token_ids=[[200, 201, 300]] * 2,
                    scheduled_spec_token_ids=scheduled)
        rows = []
        for rid in stable:
            state = p.requests[rid]
            assert state.target_batch_request_ids == tuple(stable)
            assert state.target_batch_id == f"target-verification-phase-{round_id}"
            assert 0 < state.verify_start_ns <= state.verify_end_ns
            h.verify_times.append((state.verify_start_ns, state.verify_end_ns))
            rows.append(state.committed_token_ids + (200, 201, 300))
        with h.on_rank(0):
            result = p._rank_zero_propose(ids, [len(row) for row in rows], TokenRows(rows))
        assert result["proposal_request_ids"] == stable
        assert result["draft_token_ids"] == [[200, 201, 202, 203]] * 2
        for worker in h.workers:
            owner = worker.model_runner.drafter
            identity, before = owner.identity, report(owner.identity)
            install(owner, "identity")
            assert owner.identity is identity and report(owner.identity) == before
    assert p.hooks_seen == {"verify_start": 6, "verify_end": 6}
    assert len(p.round_records) == 6
    for row in p.round_records:
        assert row["accepted_draft_token_ids"] == [200, 201]
        assert row["rejected_draft_token_ids"] == [202, 203]
        assert row["target_correction_token_ids"] == [300]
        assert row["committed_token_ids"] == [200, 201, 300]
    for rid in stable:
        assert p.requests[rid].generated_token_ids == (100,) + (200, 201, 300) * 3
        assert p.requests[rid].next_round_id == 3
        assert not p.requests[rid].finished
    saved = read_json(h.directory / "plugin-report.json")
    assert saved["hook_counts"] == p.hooks_seen and saved["round_count"] == 6
    return {"test_drive_completed": True}


@pytest.mark.parametrize("matching", ["linear", "bound-prefix"])
@pytest.mark.parametrize("prebind", [False, True])
def test_serial_install_bind_verify_and_commit(observed_startup, monkeypatch, matching, prebind):
    h = observed_startup
    h.prebind = prebind
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)
    monkeypatch.setattr(s2_runtime, "drive", lambda *a, **kw: serial_verifications(h))
    h.run("serial")
    assert h.shutdown and h.draft_shutdown
    for worker, original in zip(h.workers, h.original_maps):
        p = worker.model_runner.drafter
        assert p.internal_to_stable is p.identity.internal_to_stable
        assert original.internal_to_stable is p.identity.internal_to_stable
        assert original.stable_to_internal is p.identity.stable_to_internal
    first, second = [w.model_runner.drafter for w in h.workers]
    # Rank 1 has not bound rows (Serial transport is rank 0 only).
    assert not second.internal_to_stable
    other = list(second.definitions.values())[1]
    second.identity.bind("opaque-0", other.prompt_token_ids)
    assert first.identity.stable_id("opaque-0") != second.identity.stable_id("opaque-0")


@pytest.mark.parametrize("matching", ["linear", "bound-prefix"])
@pytest.mark.parametrize("fault,message", [
    ("unbound", "no frozen prompt binding"), ("changed", "changed stable prompt identity"),
    ("alias", "alias one stable request"), ("proposal", "initial proposal prefix mismatch"),
    ("duplicate", "no unique live proposal"),
])
def test_installed_serial_rejects_invalid_identity_and_proposal(
    observed_startup, monkeypatch, matching, fault, message
):
    h = observed_startup
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)
    monkeypatch.setattr(s2_runtime, "drive", lambda *a, **kw: serial_verifications(h, fault=fault))
    with pytest.raises(DataError if fault == "proposal" else RuntimeError, match=message):
        h.run("serial")
    assert h.shutdown


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("matching", ["linear", "bound-prefix"])
@pytest.mark.parametrize("mode", ["pingpong", "serial-split"])
def test_observed_startup_initializes_each_rank_once_before_verification(
    observed_startup, monkeypatch, probe, matching, mode
):
    h = observed_startup
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)
    # Serial-split uses the same actual worker class/startup as PingPong.
    result = h.run(point(mode)["runtime_mode"], probe=probe)
    observed = [fixed_observe.target_report(w) for w in h.workers]
    assert {r["device"]["identity"]["physical_gpu_id"] for r in observed} == {1, 2}
    for worker, original in zip(h.workers, h.original_maps):
        p = worker.model_runner.drafter
        assert report(p.identity)["mode"] == matching
        assert p.internal_to_stable is original.internal_to_stable
        assert p.stable_to_internal is original.stable_to_internal
        assert p.identity.internal_to_stable is p.internal_to_stable
        assert p.identity.stable_to_internal is p.stable_to_internal
        before = report(p.identity)
        install(p, "identity")
        assert report(p.identity) == before
        evidence = p.uuid_queries.evidence()
        assert evidence["uuid_initial_validation_count"] == 1
        assert evidence["uuid_query_mode"] == "live"
        assert evidence["uuid_verification_subprocess_query_count"] == (0 if probe else 1)
        assert evidence["uuid_cache_hit_count"] == 0
    if not probe:
        p = h.workers[0].model_runner.drafter
        rid = p.identity.stable_id("opaque-request")
        saved = read_json(h.directory / "plugin-report.json")["request_identity"]
        assert saved["bound_request_count"] == 1
        assert saved["bindings"] == [{"internal_request_id": "opaque-request", "request_id": rid}]
        states = p.state_log.read()
        assert states and all(r["internal_request_id"] == "opaque-request" for r in states)
    assert h.shutdown and result["draft_shutdown"]["shutdown"]


@pytest.mark.parametrize("matching", ["linear", "bound-prefix"])
def test_target_install_post_setup_identity_consumer(observed_startup, monkeypatch, matching):
    h = observed_startup
    monkeypatch.setenv("SR_FIXED_IDENTITY_MATCHING", matching)

    def drive(*args, **kwargs):
        monkeypatch.setenv("SR_PHASE4B2_PERFORMANCE", "1")
        for rank, worker in enumerate(h.workers):
            with h.on_rank(rank):
                p = worker.model_runner.drafter
                d = next(iter(p.definitions.values()))
                p.identity.bind("target-opaque", d.prompt_token_ids + (100,))
                # Simulated resident GPU setup; exercise the real post-setup consumer.
                p.setup_complete = True
                assert p.propose([[301]], None, None, request_ids=["target-opaque"]) == [[]]
                assert not hasattr(p, "internal_to_stable")
                assert p.identity.internal_to_stable is h.original_maps[rank].internal_to_stable
        p = h.workers[0].model_runner.drafter
        commits = [r for r in p.timing_log.read() if r.get("event") == "measured-token-commit"]
        assert len(commits) == 1 and commits[0]["request_id"] == d.request_id
        return {"test_drive_completed": True}

    monkeypatch.setattr(s2_runtime, "drive", drive)
    h.run(point("target")["runtime_mode"])
    assert h.shutdown and h.draft_shutdown
