"""Operator-only D1–D3 GPU gates. CPU CI imports/parses this module only."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from specrhythm.phase4.batched_draft_service import (
    BatchedDraftStateMachine,
    write_immutable_report,
)
from specrhythm.phase4.config import load_phase4_config
from specrhythm.phase4.draft_service import HFPersistentDraftBackend
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend, selected_draft_backend


def _proposal_row(request_id, prefix, round_id, remaining, eos):
    return {
        "request_id": request_id,
        "round_id": round_id,
        "committed_prefix_len": len(prefix),
        "committed_prefix_hash": token_prefix_hash(prefix),
        "remaining_output_budget": remaining,
        "eos_token_ids": list(eos),
    }


def _hf_fixture(config, groups, tokenizer) -> list[dict[str, Any]]:
    """One HF oracle model, destroyed before constructing the vLLM model."""
    hf = HFPersistentDraftBackend(config)
    fixtures = []
    vocab = json.loads((config.draft.resolved_model_path / "config.json").read_text())[
        "vocab_size"
    ]
    natural_eos = () if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,)
    try:
        for name, count, eos_subset in groups:
            prefixes = {
                f"{name}-{i}": tuple(
                    tokenizer.encode(
                        "Continue counting in English: "
                        + ", ".join(str(n) for n in range(1, i + 5))
                        + ","
                    )
                )
                for i in range(count)
            }
            initial = dict(prefixes)
            limits = {rid: 10 + i % 3 for i, rid in enumerate(prefixes)}
            generated = {rid: 0 for rid in prefixes}
            for rid, prefix in prefixes.items():
                hf.initialize(rid, prefix)
            forced_eos = None
            if eos_subset:
                first = next(iter(prefixes))
                hf.initialize("oracle-eos-probe", prefixes[first])
                probe, _ = hf.propose("oracle-eos-probe", 2, natural_eos)
                forced_eos = probe[-1]
                hf.finish("oracle-eos-probe")
            rounds = []
            for round_id in range(4):
                if not prefixes:
                    break
                rows, expected, sync = [], {}, []
                next_prefixes = {}
                for index, (rid, prefix) in enumerate(prefixes.items()):
                    eos = natural_eos
                    if round_id == 0 and index == 0 and forced_eos is not None:
                        eos += (forced_eos,)
                    remaining = limits[rid] - generated[rid]
                    rows.append(_proposal_row(rid, prefix, round_id, remaining, eos))
                    proposal, _ = hf.propose(rid, min(4, remaining - 1), eos)
                    expected[rid] = list(proposal)
                    terminal = proposal[-1] in eos or round_id == 3
                    if proposal[-1] in eos:
                        accepted, tail = len(proposal), ()
                    else:
                        accepted = 0 if round_id == 0 else 1 if round_id == 1 else len(proposal)
                        token = (proposal[min(accepted, len(proposal) - 1)] + 7) % vocab
                        while token in eos:
                            token = (token + 1) % vocab
                        tail = (token,)
                    delta = proposal[:accepted] + tail
                    generated[rid] += len(delta)
                    if round_id == 3 and proposal[-1] not in eos:
                        assert generated[rid] == limits[rid]
                    final = prefix + delta
                    hf.rollback(rid, accepted)
                    if tail:
                        hf.append_target_token(rid, tail[0])
                    if terminal:
                        hf.finish(rid)
                    else:
                        next_prefixes[rid] = final
                    sync.append(
                        {
                            "request_id": rid,
                            "round_id": round_id,
                            "committed_delta": list(delta),
                            "committed_prefix_hash": token_prefix_hash(final),
                            "terminal": terminal,
                        }
                    )
                rounds.append({"proposals": rows, "expected": expected, "synchronizations": sync})
                prefixes = next_prefixes
            fixtures.append(
                {
                    "name": name,
                    "initial": initial,
                    "rounds": rounds,
                    "synthetic_eos_policy_token": forced_eos,
                }
            )
        hf.torch.cuda.synchronize()
    finally:
        hf.shutdown()
    return fixtures


def run_gate(config, gate: str, count: int, output: Path, *, diagnostic: bool = False) -> dict:
    if diagnostic and gate != "D3":
        raise ValueError("materialization diagnostics are D3-only")
    if selected_draft_backend() != "vllm-batched":
        raise ValueError("D1–D3 require explicit SR_PHASE4_DRAFT_BACKEND=vllm-batched")
    if output.exists():
        raise FileExistsError("GPU gate directory must be fresh")
    output.mkdir(parents=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.draft.resolved_tokenizer_path),
        revision=config.draft.tokenizer_revision,
        trust_remote_code=config.draft.trust_remote_code,
    )
    fixtures = []
    if gate != "D1":
        groups = (
            [("single", 1, False), ("single-eos", 1, True)]
            if gate == "D2"
            else [(f"batch-{count}", count, True)]
        )
        fixtures = _hf_fixture(config, groups, tokenizer)
        write_immutable_report(output / "hf-oracle.json", {"fixtures": fixtures})
    backend = None
    observer = None
    report = {
        "schema_version": "specrhythm.phase4b3-draft-gate.v1",
        "gate": gate,
        "requested_batch_size": count,
        "valid": False,
        "errors": [],
        "synthetic_verification_fixture": gate != "D1",
        "hf_model_destroyed_before_vllm_construction": True,
        "performance_result": False,
        "comparisons": [],
        "diagnostic_only": diagnostic,
    }
    try:
        backend = VllmBatchedDraftBackend(config)
        write_immutable_report(output / "draft-startup.json", backend.provenance)
        machine = BatchedDraftStateMachine(backend)
        if gate == "D1":
            prefix = tuple(tokenizer.encode("Continue counting: 1, 2, 3,"))
            machine.initialize("construction", prefix, token_prefix_hash(prefix))
            machine.finish("construction")
        for fixture in fixtures:
            for rid, prefix in fixture["initial"].items():
                machine.initialize(rid, prefix, token_prefix_hash(prefix))
            if diagnostic:
                from specrhythm.phase4.draft_d3_diagnostics import D3MaterializationDiagnostic

                observer = D3MaterializationDiagnostic(backend)
                observer.attach()
            for round_index, item in enumerate(fixture["rounds"]):
                if observer is not None:
                    observer.round = round_index
                response = machine.batch_propose(item["proposals"])
                actual = {
                    row["request_id"]: list(row["proposal_token_ids"])
                    for row in response["proposals"]
                }
                equal = actual == item["expected"]
                report["comparisons"].append(
                    {
                        "fixture": fixture["name"],
                        "round": round_index,
                        "expected": item["expected"],
                        "actual": actual,
                        "draft_proposals_exact": equal,
                    }
                )
                if not equal:
                    raise RuntimeError("Draft proposal differs from HF oracle; gate stopped")
                committed = machine.synchronize_and_batch_propose(item["synchronizations"], [])
                for row in committed["synchronizations"]:
                    state = row["state"]
                    if state["materialized_kv_length"] != state["committed_prefix_len"]:
                        raise RuntimeError("Draft commit acknowledged an unmaterialized token")
        machine.shutdown()
        evidence = backend.report()
        if gate == "D3":
            batch = evidence["draft_batch_statistics_by_purpose"]["proposal"]
            if batch["max"] is None or batch["max"] < 2:
                raise RuntimeError("D3 did not execute an actual multi-request proposal forward")
        report["valid"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    finally:
        if observer is not None:
            try:
                observer.close()
            except Exception as error:
                report["valid"] = False
                report["errors"].append(f"diagnostic removal: {error}")
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as error:
                report["valid"] = False
                report["errors"].append(f"shutdown: {error}")
            write_immutable_report(output / "draft-backend-report.json", backend.report())
        if observer is not None:
            write_immutable_report(
                output / "draft-materialization-diagnostic.json", observer.report()
            )
        write_immutable_report(output / "gate.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", choices=("D1", "D2", "D3"), required=True)
    parser.add_argument("--request-count", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="D3-only input/page/logit observations; not performance evidence",
    )
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU execution requires the operator's explicit --allow-gpu")
    if args.gate == "D3" and args.request_count not in (2, 4, 8):
        parser.error("D3 requires --request-count 2, 4 or 8")
    if args.diagnostic and args.gate != "D3":
        parser.error("--diagnostic is D3-only")
    result = run_gate(
        load_phase4_config(args.config),
        args.gate,
        args.request_count,
        args.output,
        diagnostic=args.diagnostic,
    )
    print(json.dumps({"gate": args.gate, "valid": result["valid"], "errors": result["errors"]}))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
