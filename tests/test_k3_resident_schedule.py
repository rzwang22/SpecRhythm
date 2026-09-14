"""Production K3 -> resident -> binding -> stock; structural, not wall-clock, tests."""

import sys
from types import SimpleNamespace as NS

import pytest
from test_k3_prompt_proof import fixed_schedulers as _fixed
from test_k3_prompt_proof import s2_schedulers as _s2
from test_k3_prompt_proof import target_pool as _pool

from specrhythm.serving.k3 import MODES
from specrhythm.serving.s2_pool import publish

fixed_schedulers, s2_schedulers, target_pool = _fixed, _s2, _pool


def prepare(s, packet, path):
    from specrhythm.phase4.serial import token_prefix_hash

    packet["max_requests_per_target_forward"] = 8
    claims = []
    for i, (rid, r) in enumerate(s.requests.items()):
        r.sampling_params = NS(max_tokens=100)
        packet["requests"][rid]["cohort"] = "A" if i < 8 else "B"
        if i < 8:
            # Use production Proposal serialization, including the real current prefix.
            from specrhythm.phase4.serial import PROTOCOL_VERSION, Proposal

            proposal = Proposal(
                PROTOCOL_VERSION,
                rid,
                0,
                len(r.all_token_ids),
                token_prefix_hash(r.all_token_ids),
                (11, 12, 13),
                False,
                0,
                0,
                0,
                {},
                {},
            ).to_dict()
            claims.append(
                dict(
                    request_id=rid,
                    claim_id=rid + ":0",
                    proposal=proposal,
                    proposal_id=rid + ":p0",
                    prefix_version=0,
                    home_cohort="A",
                )
            )
    packet["pp_admission"] = dict(claims=claims, opportunity_cohort="A")
    publish(path, packet)
    s.freeze_pool()


@pytest.mark.parametrize("mode", MODES)
def test_real_scheduler_normalizes_each_current_token_once(target_pool, monkeypatch, mode):
    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    prepare(s, packet, path)
    modules = [
        sys.modules["specrhythm.phase4.resident_scheduler"],
        sys.modules["specrhythm.serving.fixed_identity"],
        sys.modules["specrhythm.phase4.request_identity"],
    ]
    calls = []
    for module in modules:

        def counted(value, *, name=module.__name__):
            calls.append((name, value))
            return int(value)

        monkeypatch.setattr(module, "int", counted, raising=False)
    before = s.s2_pool.checks
    output = s.schedule()
    binding_calls = [x for x in calls if x[0].endswith(("request_identity", "fixed_identity"))]
    # Normalization belongs to the typed, call-local identity row, only once.
    expected = sum(len(r.all_token_ids) for r in s.requests.values())
    assert len(binding_calls) == expected
    assert not [x for x in calls if x[0].endswith("fixed_identity")]
    assert s.s2_pool.checks == before + 2
    assert len(output.num_scheduled_tokens) == 8
    assert set(map(len, output.scheduled_spec_decode_tokens.values())) == {3}


@pytest.mark.parametrize("mode", MODES)
def test_old_and_new_real_schedule_have_identical_admission_semantics(
    target_pool, monkeypatch, mode
):
    import copy

    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    outputs = []
    for optimized in (False, True):
        candidate = type(s)()
        candidate.requests = copy.deepcopy(s.requests)
        candidate.running = list(candidate.requests.values())
        candidate.kv_cache_manager = s.kv_cache_manager
        candidate._resident_ready = s._resident_ready
        if not optimized:
            candidate._binding_input = lambda: lambda raw: tuple(int(t) for t in raw)
        candidate._bind_requests()
        prepare(candidate, packet, path)
        records, append = [], candidate._resident_events.append

        def record(row, records=records, append=append):
            records.append({k: v for k, v in row.items() if k != "timestamp_ns"})
            append(row)  # Real existing writer, not a dropped/idealized admission log.

        candidate._resident_events.append = record
        result = candidate.schedule()
        outputs.append(
            (
                result.num_scheduled_tokens,
                result.scheduled_spec_decode_tokens,
                candidate.s2_steps[-1]["rows"],
                {k: (v[0], v[1], v[3]) for k, v in candidate._resident_decisions.items()},
                candidate._identity().internal_to_stable.copy(),
                candidate._identity().stable_to_internal.copy(),
                records,
            )
        )
        assert len(records) == 360
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "fault",
    [
        "prompt",
        "key",
        "alias",
        "unbound",
        "suffix",
        "block",
        "frontier",
        "missing_resident",
        "stale_claim",
    ],
)
def test_real_k3_schedule_rejects_before_stock_allocation(target_pool, monkeypatch, fault):
    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", "pingpong-k3")
    prepare(s, packet, path)
    r = s.requests["0"]
    if fault == "prompt":
        r.all_token_ids[0] = 999
    elif fault == "key":
        r.request_id = "different"
    elif fault == "alias":
        s._identity().stable_to_internal["0"] = "other"
    elif fault == "unbound":
        s._identity().internal_to_stable.pop("0")
    elif fault == "suffix":
        s.requests["359"].all_token_ids.append("bad")  # Unselected generated suffix checked too.
    elif fault == "block":
        s.kv_cache_manager = NS(get_block_ids=lambda i: [[1]])
    elif fault == "frontier":
        r.num_computed_tokens = 0
    elif fault == "missing_resident":
        del s.requests["359"]
    else:
        packet["pp_admission"]["claims"][0]["proposal"]["parent_prefix_hash"] = "stale"
        publish(path, packet)
    previous = s.current_step
    with pytest.raises((ValueError, RuntimeError)):
        s.schedule()
    assert s.current_step == previous


def test_binding_type_normalization_fallback_and_history_remain_equivalent():
    from specrhythm.phase4.request_identity import FrozenPromptIdentityMap, _NormalizedTokenRow
    from specrhythm.serving.fixed_identity import BoundPromptIdentityMap

    for prompts in ({"a": [1, 2], "b": [3, 4]}, {"a": [1], "b": [1, 2]}):
        for cls in (lambda m: m, BoundPromptIdentityMap):
            old, new = [cls(FrozenPromptIdentityMap(prompts)) for _ in range(2)]
            operations = [
                ("x", [1, 2, 3]),
                ("x", [1, 2, "4", 4.5, True, -1]),
                ("x", [1, 2, "bad"]),
                ("x", [1, 2, None]),
                ("x", [1, 2, float("nan")]),
                ("x", [3, 4]),
                ("y", [1, 2]),
                ("z", [9]),
                ("z", []),
                ("", [1, 2]),
                ("x", [1, 2, 8, 9]),
            ]
            for internal, tokens in operations:
                outcomes = []
                for identity, convert in (
                    (old, lambda ts: tuple(int(t) for t in ts)),
                    (new, _NormalizedTokenRow),
                ):
                    try:
                        outcomes.append(("ok", identity.bind(internal, convert(tokens))))
                    except (ValueError, TypeError, RuntimeError, OverflowError) as e:
                        outcomes.append((type(e), str(e)))
                assert outcomes[0] == outcomes[1]
                assert old.internal_to_stable == new.internal_to_stable
                assert old.stable_to_internal == new.stable_to_internal
            # Keeping the map after removal forbids a new internal alias for retired x.
            assert new.internal_to_stable == old.internal_to_stable


def test_binding_growth_finish_refill_keeps_history_and_no_cached_token_row(
    target_pool, monkeypatch
):
    from specrhythm.phase4.request_identity import _NormalizedTokenRow

    s, _, _ = target_pool
    monkeypatch.setenv("SR_S2_MODE", "serial-k3")
    r = s.requests["0"]
    for suffix in ([1], [2, 3], [4, 5, 6]):
        r.all_token_ids.extend(suffix)
        s._bind_requests()
        assert s._identity().stable_id("0") == "0"
    r.is_finished = lambda: True
    r.all_token_ids.append("bad")  # Finished entries retain the same old skip semantics.
    s._bind_requests()
    del s.requests["0"]
    s._bind_requests()
    assert s._identity().stable_id("0") == "0"
    with pytest.raises(RuntimeError, match="alias"):
        s._identity().bind("replacement", _NormalizedTokenRow([1, 1, 10]))
    # Refill of another staged resident uses current contents, not the prior normalization.
    s.requests["359"].all_token_ids.append("bad")
    with pytest.raises(ValueError):
        s._bind_requests()


@pytest.mark.parametrize("mode", MODES)
def test_production_phase_record_serialize_export_read_and_strict_qualification(
    target_pool, monkeypatch, tmp_path, mode
):
    import copy
    import json
    import tarfile
    import time

    from specrhythm.continuation.trace import CausalTrace
    from specrhythm.serving.execution_evidence import qualify
    from specrhythm.serving.k3_resident_evidence import PHASES, resident_step, summarize
    from specrhythm.serving.ping_prepost_delivery import export

    s, packet, path = target_pool
    monkeypatch.setenv("SR_S2_MODE", mode)
    prepare(s, packet, path)
    trace = CausalTrace(True, layout="phased")
    trace.follow_control({"diagnostic_phase": "measurement"})
    # Both real wrappers write to one process-local production trace, as on the server.
    monkeypatch.setitem(type(s)._resident_span.__wrapped__.__globals__, "TRACE", trace)
    pool_cls = next(c for c in type(s).__mro__ if c.__name__ == "PoolScheduler")
    monkeypatch.setitem(pool_cls.schedule.__globals__, "TRACE", trace)
    start = time.monotonic_ns()
    s.schedule()
    end = time.monotonic_ns()
    step = dict(s.s2_steps[-1], start_ns=start, end_ns=end, window=True)
    runtime = dict(
        point=dict(mode=mode), target_steps=[step], host=dict(causal_timeline=trace.report())
    )
    raw = tmp_path / "runtime.json"
    raw.write_text(json.dumps(runtime))
    e = resident_step(step, runtime["host"]["causal_timeline"]["rows"])
    assert e["status"] == "COMPLETE", e["errors"]
    assert e["work"]["binding"] == dict(
        live_requests=360, normalized_rows=360, binding_token_visits=1440
    )
    assert e["work"]["admission_records"]["admission_records"] == 360
    assert set(e["phases_ms"]) == set(PHASES)
    report = summarize([dict(step_index=0, resident_schedule=e)])
    assert report["status"] == "COMPLETE"

    def qualified_scope(value):
        # Same formal diagnostic qualifier as production. GPU evidence is absent in
        # this CPU scheduler harness and remains failed; it is never fabricated.
        audit = dict(mode=mode, pingpong=dict(
            pipeline=dict(dispatch=dict(resident_schedule=value))))
        result = qualify(audit)
        assert result["diagnostic_integrity"] == "FAILED"
        return [e for e in result["errors"] if e.startswith("K3 resident")]

    assert not qualified_scope(report)
    archive = tmp_path / "delivery.tar.gz"
    inv = export(tmp_path, archive, first_code=23, modes=MODES)
    # This scheduler-only harness has no GPU startup: exporter must not invent it.
    assert inv["export_status"] == "INCOMPLETE"
    with tarfile.open(archive) as t:
        paths = json.load(t.extractfile("inventory.json"))["logical_paths"]
        reread = json.load(t.extractfile(paths["runtime.json"]))
    assert resident_step(reread["target_steps"][0], reread["host"]["causal_timeline"]["rows"]) == e
    replayed = summarize([dict(step_index=0, resident_schedule=resident_step(
        reread["target_steps"][0], reread["host"]["causal_timeline"]["rows"]))])
    assert not qualified_scope(replayed)
    assert trace.report()["dropped_rows"] == 0
    for field in ("binding_token_visits", "normalized_rows"):
        rows = copy.deepcopy(runtime["host"]["causal_timeline"]["rows"])
        binding = next(r for r in rows if r["category"] == "target_resident_binding")
        binding["work"].pop(field)
        broken = resident_step(step, rows)
        rejected = summarize([dict(step_index=0, resident_schedule=broken)])
        assert rejected["status"] == "INCOMPLETE" and rejected["work_totals"] is None
        assert qualified_scope(rejected)
    rows = [
        r
        for r in runtime["host"]["causal_timeline"]["rows"]
        if r["category"] != "target_resident_stock"
    ]
    assert resident_step(step, rows)["phases_ms"]["stock"] is None
