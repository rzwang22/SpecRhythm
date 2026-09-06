from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_serial import phase4_config as _config_fixture
from test_phase4_vllm_draft import FakeWorker

from specrhythm.phase4.batched_draft_service import BatchedDraftStateMachine
from specrhythm.phase4.draft_batch import DraftMaterialization
from specrhythm.phase4.draft_d3_diagnostics import (
    D3MaterializationDiagnostic,
    assert_private_blocks,
    assert_runner_rows,
    authoritative_logit_ids,
)
from specrhythm.phase4.draft_gate import run_gate
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend

phase4_config = _config_fixture
FIXTURE = Path(__file__).parent / "fixtures/phase4b3/d3-b8-8709c81.json"


def test_real_artifact_temporal_and_block_boundary_localization():
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["fixtures"][0]
    assert [len(p) for p in rows["initial"].values()] == [17, 20, 23, 26, 29, 32, 36, 40]
    assert [len(r["proposals"]) for r in rows["rounds"]] == [8, 7, 7, 7]
    assert [o["draft_proposals_exact"] for o in fixture["observed"]] == [True, True, False]
    expected, actual = fixture["observed"][-1]["expected"], fixture["observed"][-1]["actual"]
    assert [rid for rid in expected if expected[rid] != actual[rid]] == ["batch-8-7"]
    assert actual["batch-8-7"] == expected["batch-8-1"] == [11, 227, 11, 227]
    assert expected["batch-8-7"] == [11, 16, 17, 11]
    parents = {r["request_id"]: r["committed_prefix_len"] for r in rows["rounds"][2]["proposals"]}
    assert parents["batch-8-7"] == 43
    assert (43 + 15) // 16 == (44 + 15) // 16 == 3
    assert [rid for rid, n in parents.items() if n % 16 == 0] == ["batch-8-4"]


def test_b8_eos_to_b7_partial_correction_repeated_rounds_preserve_context(phase4_config):
    """Replay actual prefixes/acceptance with an independent CPU KV array oracle.

    This guards the transition pattern; it does NOT reproduce or explain the
    observed GPU mismatch. A later root-cause regression must fail the old code.
    """
    fixture = json.loads(FIXTURE.read_text())["fixtures"][0]
    prefixes = {rid: tuple(tokens) for rid, tokens in fixture["initial"].items()}
    next_tokens = {}
    for item in fixture["rounds"]:
        for rid, tokens in item["expected"].items():
            for step, token in enumerate(tokens):
                next_tokens[prefixes[rid] + tuple(tokens[:step])] = token
        for row in item["synchronizations"]:
            prefixes[row["request_id"]] += tuple(row["committed_delta"])

    class PhysicalOracle(FakeWorker):
        def materialize(self, rows, purpose):
            super().materialize(rows, purpose)  # Checks own KV below the frontier.
            return {
                row.request_id: next_tokens.get(
                    tuple(self.memory[row.request_id][: len(row.context)]), 0
                )
                for row in reversed(rows)
            }

    worker = PhysicalOracle()
    backend = VllmBatchedDraftBackend(phase4_config, worker=worker)
    machine = BatchedDraftStateMachine(backend)
    for rid, tokens in fixture["initial"].items():
        machine.initialize(rid, tokens, token_prefix_hash(tokens))
    for n, item in enumerate(fixture["rounds"]):
        response = machine.batch_propose(item["proposals"])
        actual = {r["request_id"]: r["proposal_token_ids"] for r in response["proposals"]}
        assert actual == item["expected"]
        if n == 2:
            assert actual["batch-8-7"] == [11, 16, 17, 11]
            assert actual["batch-8-7"] != actual["batch-8-1"]
            assert "sr-draft:batch-8-0" not in worker.memory
        machine.synchronize_and_batch_propose(item["synchronizations"], [])
    machine.shutdown()
    assert worker.memory == {}
    assert backend.report()["draft_batch_statistics_by_purpose"]["proposal"]["max"] == 8


@pytest.mark.parametrize(
    "owners",
    [
        {"r1": [[1, 2]], "r7": [[2, 3]]},
        {"r1": [[1, 1]]},
        {"r1": [[-1]]},
    ],
)
def test_private_page_alias_and_bad_identity_fail_closed(owners):
    with pytest.raises(RuntimeError):
        assert_private_blocks(owners)


def test_group_id_is_part_of_physical_page_identity():
    assert_private_blocks({"r1": [[1], [2]], "r7": [[2], [1]]})


@pytest.mark.parametrize("indices", [[0, 1], [1, 1], [1, 2, 3]])
def test_equal_request_id_sets_do_not_prove_logits_row_domain(indices):
    with pytest.raises(RuntimeError, match="row domain"):
        authoritative_logit_ids(["r7", "r1"], [0, 2, 3], indices)
    assert authoritative_logit_ids(["r7", "r1"], [0, 2, 3], [1, 2]) == ("r7", "r1")


def prepared_rows():
    plans = {
        "r1": DraftMaterialization("r1", (1, 2, 11), 2),
        "r7": DraftMaterialization("r7", (8, 9, 10, 11), 3),
    }
    owners = {"r1": [[2]], "r7": [[7]]}
    rows = [
        {
            "internal_id": rid,
            "context_sha256": token_prefix_hash(p.context),
            "runner_context_sha256": token_prefix_hash(p.context),
            "computed_tokens": p.valid_length,
            "runner_computed_tokens": p.valid_length,
            "prompt_tokens": len(p.context),
            "runner_prompt_tokens": len(p.context),
            "active_token_count": len(p.context),
            "scheduled_tokens": 1,
            "scheduled_input_token_ids": [11],
            "block_ids": owners[rid],
        }
        for rid, p in reversed(list(plans.items()))
    ]
    return plans, rows, owners


@pytest.mark.parametrize(
    "field",
    [
        "context_sha256",
        "runner_context_sha256",
        "computed_tokens",
        "runner_computed_tokens",
        "prompt_tokens",
        "runner_prompt_tokens",
        "active_token_count",
        "block_ids",
    ],
)
def test_request7_cannot_receive_request1_context_frontier_or_blocks(field):
    plans, rows, owners = prepared_rows()
    assert_runner_rows(plans, rows, owners)  # Deliberately reversed actual row order.
    rows[0][field] = copy.deepcopy(rows[1][field])
    with pytest.raises(RuntimeError, match="r7"):
        assert_runner_rows(plans, rows, owners)


def test_no_diagnostic_entry_to_serving_or_non_d3_gate(tmp_path):
    with pytest.raises(ValueError, match="D3-only"):
        run_gate(None, "D2", 1, tmp_path, diagnostic=True)
    for name in (
        "draft_service.py",
        "dual_service.py",
        "vllm_draft_backend.py",
        "vllm_draft_worker.py",
    ):
        text = (Path("src/specrhythm/phase4") / name).read_text()
        assert "D3MaterializationDiagnostic" not in text
        assert "draft_d3_diagnostics" not in text


def test_observer_records_failure_and_restores_only_its_instance_methods():
    # No CUDA import, no model execution. Exercise the actual attach/finally path.
    model = SimpleNamespace(compute_logits=lambda x: x)
    removed = []
    model.register_forward_pre_hook = lambda fn: SimpleNamespace(
        remove=lambda: removed.append(True)
    )
    runner = SimpleNamespace(_update_states=lambda x: None, _prepare_inputs=lambda *a: None)
    worker = SimpleNamespace(
        runner=runner,
        model=model,
        views={"r1": 1, "r7": 2},
        kv=SimpleNamespace(get_block_ids=lambda rid: ([4],)),
        materialize=lambda rows, purpose: pytest.fail("alias must stop before execution"),
    )
    original = worker.materialize
    observer = D3MaterializationDiagnostic(SimpleNamespace(worker=worker, states={}))
    observer.attach()
    try:
        with pytest.raises(RuntimeError, match="KV alias"):
            worker.materialize([DraftMaterialization("r7", (1, 2), 1)], "proposal")
    finally:
        observer.close()
    assert worker.materialize is original
    assert removed == [True]
    report = observer.report()
    assert not report["all_structural_checks_passed"]
    assert report["frames"][0]["last_external_owners"] == {"r1": [[4]], "r7": [[4]]}
    assert report["observer_removed"] and report["performance_result"] is False
    json.dumps(report)


class Tensor:
    """Small CPU stand-in for the observer's read-only tensor protocol."""

    def __init__(self, data):
        self.data = data
        self.shape = (
            (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
        )

    def __getitem__(self, key):
        if isinstance(key, tuple):
            first, second = key
            value = self.data[first][second]
        else:
            value = self.data[key]
        return Tensor(value) if isinstance(value, list) else value

    def tolist(self):
        return copy.deepcopy(self.data)

    def detach(self):
        return self

    cpu = float = detach

    def argmax(self, dim=-1):
        return Tensor([max(range(len(row)), key=row.__getitem__) for row in self.data])

    def topk(self, k, dim=-1):
        # Reverse index ties deliberately differ from argmax's first index.
        indices = [
            sorted(range(len(row)), key=lambda i: (row[i], i), reverse=True)[:k]
            for row in self.data
        ]
        return Tensor([[r[i] for i in indices[n]] for n, r in enumerate(self.data)]), Tensor(
            indices
        )


@pytest.mark.parametrize(
    "fault", [None, "gpu-page-alias", "completion-row-mismatch", "logits-indices"]
)
def test_all_observer_seams_capture_actual_row_order_without_mutation(fault):
    plans, _, owners = prepared_rows()
    ids = ["r7", "r1"]
    page_rows = [[7], [2]]
    table = SimpleNamespace(
        blocks_per_kv_block=1,
        block_size=16,
        num_blocks_per_row=[1, 1],
        get_numpy_array=lambda: Tensor(page_rows),
        get_device_tensor=lambda n: Tensor([[2], [2]] if fault == "gpu-page-alias" else page_rows),
        slot_mapping=SimpleNamespace(gpu=Tensor([115, 34])),
    )
    batch = SimpleNamespace(
        req_ids=ids,
        num_reqs=2,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        num_computed_tokens_cpu=[3, 2],
        num_prompt_tokens=[4, 3],
        token_ids_cpu=Tensor([[8, 9, 10, 11], [1, 2, 11]]),
        block_table=SimpleNamespace(block_tables=[table]),
        _get_active_token_count=lambda i: [4, 3][i],
    )
    runner = SimpleNamespace(
        input_batch=batch,
        requests={
            r: SimpleNamespace(
                num_computed_tokens=p.valid_length,
                num_prompt_tokens=len(p.context),
                prompt_token_ids=list(p.context),
            )
            for r, p in plans.items()
        },
        query_start_loc=SimpleNamespace(gpu=Tensor([0, 1, 2])),
        input_ids=SimpleNamespace(gpu=Tensor([11, 11])),
        positions=Tensor([3, 2]),
        num_computed_tokens=Tensor([3, 2]),
        seq_lens=Tensor([4, 3]),
        req_indices=SimpleNamespace(gpu=Tensor([0, 1])),
        _update_states=lambda output: "unchanged-update-result",
        _prepare_inputs=lambda output, tokens: (
            Tensor([1, 0] if fault == "logits-indices" else [0, 1]),
            None,
        ),
    )
    logits = Tensor([[0.0, 5.0, 5.0], [1.0, 0.0, 3.0]])
    hooks = []
    model = SimpleNamespace(compute_logits=lambda hidden: logits)

    def register(hook):
        hooks.append(hook)
        return SimpleNamespace(remove=lambda: hooks.remove(hook))

    model.register_forward_pre_hook = register

    def materialize(rows, purpose):
        output = SimpleNamespace(num_scheduled_tokens={r.request_id: len(r.suffix) for r in rows})
        assert runner._update_states(output) == "unchanged-update-result"
        runner._prepare_inputs(output, [1, 1])
        for hook in hooks:
            hook(model, ())
        assert model.compute_logits(None) is logits
        return {"r1": logits[1], "r7": logits[1 if fault == "completion-row-mismatch" else 0]}

    worker = SimpleNamespace(
        runner=runner,
        model=model,
        views={r: 1 for r in ids},
        kv=SimpleNamespace(get_block_ids=lambda rid: owners[rid]),
        materialize=materialize,
        torch=SimpleNamespace(stack=lambda values: Tensor([v.tolist() for v in values])),
    )
    states = {r: SimpleNamespace(internal_id=r) for r in ids}
    observer = D3MaterializationDiagnostic(SimpleNamespace(worker=worker, states=states))
    observer.round = 2
    observer.attach()
    try:
        if fault:
            with pytest.raises(RuntimeError, match="GPU input|wrong IDs|row domain"):
                worker.materialize(list(plans.values()), "proposal")
        else:
            result = worker.materialize(list(plans.values()), "proposal")
            assert result["r7"].tolist() == logits[0].tolist()
    finally:
        observer.close()
    assert worker.materialize is materialize and hooks == []
    assert batch.req_ids == ids and table.get_numpy_array().tolist() == page_rows
    frame = observer.report()["frames"][0]
    assert frame["round"] == 2 and frame["checks_passed"] is (fault is None)
    assert frame["after_update_states"][0]["internal_id"] == "r7"
    if fault == "logits-indices":
        assert frame["prepared_logits_domain"]["logits_indices"] == [1, 0]
        assert "immediately_before_model" not in frame
    else:
        assert len(frame["immediately_before_model"]) == 2
    if fault == "completion-row-mismatch":
        assert frame["completion_mapping"]["top1_by_id"] == {"r1": 2, "r7": 2}
    if not fault:
        assert frame["after_forward"]["authoritative_logits_request_ids"] == ids
        assert frame["after_forward"]["top1_token_ids"] == [1, 2]
        assert frame["after_forward"]["top2_token_ids"][0][0] == 2  # Tie is not a false error.
    json.dumps(observer.report())


def test_gate_retains_diagnostic_failure_once_and_shuts_down(monkeypatch, tmp_path):
    from specrhythm.phase4 import draft_d3_diagnostics, draft_gate

    class Backend:
        provenance = {"cpu_test_double": True}

        def initialize(self, request_id, prefix):
            return {}

        def shutdown(self):
            self.closed = True

        def report(self):
            return {"closed": self.closed}

    class Observer:
        def attach(self):
            raise RuntimeError("injected diagnostic attachment failure")

        def close(self):
            self.removed = True

        def report(self):
            return {"observer_removed": self.removed, "diagnostic_only": True}

    backend, observer = Backend(), Observer()
    fixture = {"name": "cpu-failure", "initial": {}, "rounds": []}
    monkeypatch.setenv("SR_PHASE4_DRAFT_BACKEND", "vllm-batched")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)),
    )
    monkeypatch.setattr(draft_gate, "_hf_fixture", lambda *args: [fixture])
    monkeypatch.setattr(draft_gate, "VllmBatchedDraftBackend", lambda config: backend)
    monkeypatch.setattr(draft_d3_diagnostics, "D3MaterializationDiagnostic", lambda b: observer)
    config = SimpleNamespace(
        draft=SimpleNamespace(
            resolved_tokenizer_path=tmp_path,
            tokenizer_revision="test",
            trust_remote_code=False,
        )
    )
    output = tmp_path / "D3-B8"
    report = run_gate(config, "D3", 8, output, diagnostic=True)
    assert not report["valid"] and report["diagnostic_only"]
    assert "injected diagnostic attachment failure" in report["errors"][0]
    assert json.loads((output / "draft-backend-report.json").read_text()) == {"closed": True}
    path = output / "draft-materialization-diagnostic.json"
    original = path.read_bytes()
    assert json.loads(original) == {"observer_removed": True, "diagnostic_only": True}
    with pytest.raises(FileExistsError, match="fresh"):
        run_gate(config, "D3", 8, output, diagnostic=True)
    assert path.read_bytes() == original
