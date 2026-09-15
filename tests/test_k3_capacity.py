"""Real fixed run capacity arithmetic; CUDA creation/queries alone are substituted."""

from types import SimpleNamespace as NS

import pytest
from test_serving_s1 import request
from test_serving_s2 import ranks

from specrhythm.serving import fixed_runtime
from specrhythm.serving.common import DataError
from specrhythm.serving.fixed_plan import capacity_metadata
from specrhythm.serving.k3 import MODES
from specrhythm.serving.k3_capacity import check, preflight, reservation
from specrhythm.serving.s1_workload import write_once
from specrhythm.serving.s2_plan import capacity_for


@pytest.fixture
def capacity_run(tmp_path, monkeypatch):
    def build(mode):
        metadata = capacity_metadata(mode, active_limit=16, resident_requirement=360,
                                     target_sequence_limit=512)
        raw = [{**r, "mode": mode} for r in ranks()[:3]]
        raw.sort(key=lambda r: (r["role"] == "draft", r["physical_gpu_id"]))
        assert [r["role"] for r in raw] == ["target", "target", "draft"]
        manifest = dict(active_limit=16, fixed_diagnostic=dict(
            capacity={mode: metadata}, options=dict(
            drain_timeout=60)), sha256="execution", workload_sha256="workload")
        cfg = NS(
            scheduler_config=NS(async_scheduling=False, max_num_seqs=512,
                                max_num_batched_tokens=4096),
            model_config=NS(max_model_len=4096), cache_config=NS(enable_prefix_caching=False),
            speculative_config=NS(num_speculative_tokens=3),
        )
        calls = []
        llm = NS(llm_engine=NS(vllm_config=cfg, engine_core=NS(
            shutdown=lambda **kw: calls.append("Target shutdown"))),
            collective_rpc=lambda *a, **kw: [
                dict(s2_capacity=r, batch_invariant_effective=True, s1_effective_capacity={})
                for r in raw[:2]
            ])
        rows = [request(i, 512) for i in range(360)]
        monkeypatch.setattr(fixed_runtime, "configure", lambda *a: (
            NS(target=None, logprobs=1), manifest, rows))
        monkeypatch.setattr(fixed_runtime, "make_engine", lambda *a, **kw: llm)
        monkeypatch.setattr(fixed_runtime, "validate_worker_ranks", lambda *a: [])
        monkeypatch.setattr(fixed_runtime, "client_for", lambda *a: NS(
            call=lambda *a: dict(shutdown=True)))
        def drive(*a, **kw):
            raise DataError("entered drive after real capacity checks")
        monkeypatch.setattr(fixed_runtime, "drive", drive)
        write_once(tmp_path / "draft-startup.json", dict(s2_capacity=raw[2],
            fixed_engine_limits=dict(max_num_seqs=128, max_num_batched_tokens=4096)))
        return NS(mode=mode, point=dict(mode=mode, runtime_mode=mode, scan=True, probe=True,
                  batch=16), calls=calls, raw=raw, rows=rows, metadata=metadata, llm=llm)
    return build


@pytest.mark.parametrize("mode", MODES)
def test_production_k3_capacity_reaches_drive_with_real_capacity_function(
    mode, capacity_run, tmp_path
):
    h = capacity_run(mode)
    # This first failed on the actual old production call with speculative_tokens=3.
    # No substitution of capacity_for or preconstructed capacity PASS report.
    with pytest.raises(DataError, match="entered drive after real capacity checks"):
        fixed_runtime.run(tmp_path, None, tmp_path, h.point, probe=True)
    assert h.calls == ["Target shutdown"]
    from specrhythm.serving.common import read_json

    actual = read_json(tmp_path / "actual-capacity.json")
    assert [r["reserved_speculative_positions"] for r in actual["checks"]] == (
        [4, 4, 6] if "eager" in mode else [4, 4, 4])
    assert all(r["candidate_length"] == 3 for r in actual["checks"])
    assert h.llm.llm_engine.vllm_config.speculative_config.num_speculative_tokens == 3


def test_legacy_capacity_still_refuses_three_positions():
    with pytest.raises(DataError, match="baseline K4 reserve"):
        capacity_for([request(0, 32)], ranks()[0], speculative_tokens=3)


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("prompt", [15, 16, 17])
@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_capacity_rounding_exact_fit_and_deficit(role, prompt, maximum):
    rows = [NS(request_id="arithmetic", prompt_length=prompt, maximum_new_tokens=maximum)]
    mode = MODES[-1]
    rank = next({**r, "mode": mode, "num_gpu_blocks": 64} for r in ranks() if r["role"] == role)
    meta = capacity_metadata(mode, active_limit=16)
    def evaluate(r):
        return check(rows, r, mode=mode, active_limit=16, metadata=meta)
    value = evaluate(rank)
    reserve = reservation(mode, role)["reserved_speculative_positions"]
    assert value["required_blocks"] == (prompt + maximum + reserve + 15) // 16 + 1 + 32
    assert evaluate({**rank, "num_gpu_blocks": value["required_blocks"]})["valid"]
    assert not evaluate({**rank, "num_gpu_blocks": value["required_blocks"] - 1})["valid"]
    assert not evaluate({**rank, "free_memory_bytes": 0})["valid"]
    old = capacity_for(rows, {**rank, "mode": "serial"}, active_limit=16)
    assert "speculative_capacity_tokens" not in old and "extra_speculative_tokens" not in old
    assert old["required_blocks"] == (prompt + maximum + 4 + 15) // 16 + 1 + 32


@pytest.mark.parametrize("field", [
    "block_size", "num_gpu_blocks", "free_memory_bytes", "vocab_size",
])
@pytest.mark.parametrize("value", [True, 3.0, -1, None])
def test_raw_capacity_types_never_coerced(field, value):
    mode = MODES[0]
    raw = {**ranks()[0], "mode": mode, field: value}
    with pytest.raises(DataError, match="raw rank capacity"):
        check([request(0, 32)], raw, mode=mode, active_limit=16,
              metadata=capacity_metadata(mode, active_limit=16))


@pytest.mark.parametrize("value", [True, 3.0, -1, None])
def test_candidate_configuration_invalid_or_missing(value, monkeypatch):
    from specrhythm.serving.k3 import PARAMETERS

    if value is None:
        monkeypatch.delitem(PARAMETERS, "candidate_length")
    else:
        monkeypatch.setitem(PARAMETERS, "candidate_length", value)
    with pytest.raises(DataError, match="explicit integer"):
        preflight()


def test_static_contract_is_not_physical_capacity():
    result = preflight()
    assert result["static_contract"] == "PASS" and result["GPU_capacity"] == "PENDING"
    assert len(result["reservations"]) == 8
    assert [r["required_speculative_positions"] for r in result["reservations"]] == [
        3,
        3,
        3,
        6,
        3,
        3,
        3,
        6,
    ]
    assert [r["reserved_speculative_positions"] for r in result["reservations"]] == [
        4,
        4,
        4,
        6,
        4,
        4,
        4,
        6,
    ]


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("early_steps", [0, 1, 3])
@pytest.mark.parametrize("reject_at", [None, 0, 1])
def test_live_kv_and_correction_fit_declared_positions(eager, early_steps, reject_at):
    from test_k3 import machine
    from test_ping_prepost import admit, run_work
    from test_prepost_protocol import cleanup
    from test_serial_eager_owner import verify_row

    from specrhythm.phase4.serial import token_prefix_hash

    m = machine(eager, ids=("a",))
    required = reservation(MODES[-1] if eager else MODES[0], "draft")[
        "required_speculative_positions"]
    try:
        p = admit(m)["claims"][0]["proposal"]
        state = m.backend.states["a"]
        prefix = state.prefix
        assert len(state.proposal) == 3 and state.materialized == len(prefix) + 2
        m.verify_start([verify_row(p)])
        for _ in range(early_steps):
            m.step()
            generated = sum(len(j.generated) for j in m.backend.prepost_jobs.values())
            assert 3 + generated <= required
            assert state.materialized == len(prefix) + 2 + generated
        delta = tuple(p["proposal_token_ids"])
        if reject_at is not None:
            delta = delta[:reject_at] + (999,)
        final = prefix + delta
        start = len(m.backend.prepost_forwards.rows())
        m.feedback([dict(request_id="a", round_id=p["round_id"], committed_delta=list(delta),
                         committed_prefix_hash=token_prefix_hash(final), terminal=False)])
        run_work(m)
        assert state.prefix == final and len(state.proposal) == 3
        assert state.materialized == len(final) + 2
        forwards = m.backend.prepost_forwards.rows()[start:]
        repair = [r for r in forwards if r["purpose"] == "prepost_post"]
        if reject_at is not None or not eager:
            # Fenced rollback/catch-up materializes just the final committed token;
            # its logits seed K3, then TWO extensions. No extra correction reserve.
            assert len(repair) == 1 and repair[0]["materialized_positions"] == [1]
            assert sum(r["purpose"] == "prepost_extension" for r in forwards) == 2
        else:
            assert not repair  # Full reuse must not add candidate four.
    finally:
        cleanup(m)


def test_no_gpu_static_entrypoint(tmp_path):
    import subprocess
    import sys

    from specrhythm.serving.common import read_json

    path = tmp_path / "static.json"
    result = subprocess.run([sys.executable, "-m", "specrhythm.serving.k3_capacity",
                             "--output", str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert read_json(path) == preflight()
