"""Execute pinned stock CPU bookkeeping, then opt-in projection and publication."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from specrhythm.serving.prepost_target import install


class Array(list):
    @property
    def shape(self):
        return (len(self), len(self[0])) if self and isinstance(self[0], list) else (len(self),)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            value = super().__getitem__(row)[column]
            return Array(value) if isinstance(column, slice) else value
        result = super().__getitem__(key)
        return Array(result) if isinstance(key, slice) else result

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, column = key
            if isinstance(column, slice) and not isinstance(value, list):
                value = [value] * len(self[row][column])
            self[row][column] = value
        else:
            super().__setitem__(key, value)

    def tolist(self):
        return list(self)


np = NS(
    array=Array,
    zeros=lambda shape, dtype: Array([[0] * shape[1] for _ in range(shape[0])]),
    ones=lambda shape, dtype: Array([[True] * shape[1] for _ in range(shape[0])]),
    nonzero=lambda values: (Array(i for i, v in enumerate(values) if v),),
)


@pytest.fixture
def runner_class():
    root = Path(os.environ.get("SR_PHASE4B3_SOURCE_AUDIT", "/tmp/rolling-eager-vllm-source"))
    path = root / "vllm/v1/worker/gpu_model_runner.py"
    if not path.exists():
        pytest.skip("pinned vLLM source audit unavailable")
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_bookkeeping_sync"
    )
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    namespace = {
        "np": np,
        "envs": NS(VLLM_COMPUTE_NANS_IN_LOGITS=False),
        "RejectionSampler": NS(
            parse_output=lambda values, *a, **kw: (
                [[int(t) for t in row if t >= 0] for row in values],
                kw["logprobs_tensors"],
            )
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return type("StockCPUBookkeeping", (), {"_bookkeeping_sync": namespace["_bookkeeping_sync"]})


def make_runner(cls):
    r = cls()
    r.use_async_scheduling = False
    r.routed_experts_initialized = False
    r.max_model_len = 32
    r.discard_request_mask = NS(np=np.array([False, False]))
    r.drafter = NS(eos_token_ids=(99,))
    r.requests = {
        rid: NS(output_token_ids=[1], sampling_params=NS(max_tokens=30)) for rid in ("a", "b")
    }
    r.input_batch = NS(
        num_reqs=2,
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        num_tokens_no_spec=np.array([3, 3]),
        token_ids_cpu=np.zeros((2, 32), int),
        is_token_ids=np.ones((2, 32), bool),
        vocab_size=100,
        generators={},
    )
    r._get_prompt_logprobs_dict = lambda *a: {}
    return r


@pytest.mark.parametrize("reject", [True, False])
def test_stock_bookkeeping_mixed_rows_published_without_bonus_or_padding(runner_class, reject):
    runner = make_runner(runner_class)
    install(runner)
    original = runner._bookkeeping_sync
    install(runner)
    assert runner._bookkeeping_sync is original
    scheduler = NS(
        scheduled_spec_decode_tokens={"a": [2, 3, 4, 5], "b": [6]},
        num_scheduled_tokens={"a": 5, "b": 2},
    )
    sampled = np.array([[2, 3, 4, 5, 7], [8 if reject else 6, -1 if reject else 9, -1, -1, -1]])
    # Real parser owns -1 removal; wrapper rejects any remaining negative ID.
    logprobs = NS(cu_num_generated_tokens=[0, 5, 6 if reject else 7])
    result = runner._bookkeeping_sync(
        scheduler, NS(sampled_token_ids=sampled, logprobs_tensors=logprobs), None, [0] * 7, 7
    )
    assert result[2] == [[2, 3, 4, 5], [8 if reject else 6]]
    assert result[1] is logprobs  # Original row offsets remain correct for canonical slice counts.
    assert runner.input_batch.num_tokens_no_spec.tolist() == [7, 4]
    assert runner.requests["a"].output_token_ids == [1, 2, 3, 4, 5]
    assert runner.requests["b"].output_token_ids == [1, 8 if reject else 6]
    event = runner.prepost_samples.rows()[0]
    assert event["actual_candidate_lengths"] == [4, 1]
    assert event["target_effective_query_positions"] == {"a": 5, "b": 2}
    assert not runner.input_batch.is_token_ids[0, 7]
    assert result[2][0][-1] != 7  # The unused Target bonus never becomes output or candidate.


def test_stock_bookkeeping_eos_and_output_limit(runner_class):
    runner = make_runner(runner_class)
    runner.requests["a"].sampling_params.max_tokens = 3
    install(runner)
    result = runner._bookkeeping_sync(
        NS(
            scheduled_spec_decode_tokens={"a": [2, 3, 4, 5], "b": [99]},
            num_scheduled_tokens={"a": 5, "b": 2},
        ),
        NS(
            sampled_token_ids=np.array([[2, 3, 4, 5, 7], [99, 8, -1, -1, -1]]),
            logprobs_tensors=None,
        ),
        None,
        [0] * 7,
        7,
    )
    assert result[2] == [[2, 3], [99]]
    assert all(row["terminal"] for row in runner.prepost_samples.rows()[0]["requests"])


def test_actual_stock_rejection_frontier_retains_one_uncached_root():
    root = Path(os.environ.get("SR_PHASE4B3_SOURCE_AUDIT", "/tmp/rolling-eager-vllm-source"))
    path = root / "vllm/v1/core/sched/scheduler.py"
    if not path.exists():
        pytest.skip("pinned vLLM source audit unavailable")
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "update_from_output"
    )
    # Execute the actual stock rejection/frontier block, not a mirrored formula.
    block = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("scheduled_spec_token_ids and")
    )
    for proposal, delta, canonical_count in (
        ([2], [2], 1),
        ([2, 3, 4, 5], [2, 3, 4, 5], 4),
        ([2, 3, 4, 5], [2, 9], 2),
    ):
        request = NS(
            async_tokens_to_discard=0,
            num_computed_tokens=100 + len(proposal),
            num_output_placeholders=0,
        )
        ns = dict(
            scheduled_spec_token_ids=proposal,
            generated_token_ids=delta,
            request=request,
            self=NS(num_sampled_tokens_per_step=1, make_spec_decoding_stats=lambda *a, **k: None),
            spec_decoding_stats=None,
            scheduler_output=NS(num_invalid_spec_tokens={}),
            req_id="a",
        )
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), ns)
        assert request.num_computed_tokens == 100 + canonical_count - 1
        # This stock counter is a cache-frontier count, not our no-bonus accepted count.
        assert ns["num_accepted"] == canonical_count - 1


@pytest.mark.parametrize("lengths", [[1, 4] * 8, [4, 1] * 8])
def test_actual_stock_ragged_sampling_indices_are_one_b16_batch(lengths):
    root = Path(os.environ.get("SR_PHASE4B3_SOURCE_AUDIT", "/tmp/rolling-eager-vllm-source"))
    path = root / "vllm/v1/worker/gpu_model_runner.py"
    if not path.exists():
        pytest.skip("pinned vLLM source audit unavailable")
    function = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_calc_spec_decode_metadata"
    )
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None

    class Vector(Array):
        def __add__(self, other):
            return (
                Vector(a + b for a, b in zip(self, other))
                if isinstance(other, list)
                else Vector(a + other for a in self)
            )

        def __sub__(self, other):
            return (
                Vector(a - b for a, b in zip(self, other))
                if isinstance(other, list)
                else Vector(a - other for a in self)
            )

        def __iadd__(self, other):
            return self + other

        def __getitem__(self, key):
            if isinstance(key, list):
                return Vector(self[i] for i in key)
            result = super().__getitem__(key)
            return Vector(result) if isinstance(key, slice) else result

    def cumsum(values, scratch, **kwargs):
        total, ends, ranges = 0, [], []
        for n in values:
            total += n
            ends.append(total)
            ranges.extend(range(n))
        scratch[: len(ranges)] = ranges
        return Vector(ends)

    namespace = dict(
        np=NS(
            int32=int,
            repeat=lambda rows, counts: Vector(v for v, n in zip(rows, counts) for _ in range(n)),
        ),
        async_tensor_h2d=lambda values, **kwargs: values,
        SpecDecodeMetadata=lambda **kwargs: NS(**kwargs),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    contexts = [[1000 + i] + [10 * i + j for j in range(n)] for i, n in enumerate(lengths)]
    r = NS(
        device="CPU test boundary",
        _arange_scratch=Vector([0] * 80),
        input_ids=NS(gpu=Vector(t for row in contexts for t in row)),
        _get_cumsum_and_arange=cumsum,
    )
    cumulative = cumsum([n + 1 for n in lengths], Vector([0] * 80))
    result = namespace["_calc_spec_decode_metadata"](r, Vector(lengths), cumulative)
    assert result.num_draft_tokens == lengths
    assert result.logits_indices == list(range(56))
    assert result.draft_token_ids == [t for row in contexts for t in row[1:]]
    assert len(result.target_logits_indices) == 40
    assert len(result.bonus_logits_indices) == 16
    assert set(result.target_logits_indices).isdisjoint(result.bonus_logits_indices)
