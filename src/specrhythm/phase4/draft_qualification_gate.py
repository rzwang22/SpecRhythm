"""Operator-only D3 v2: production proposals, ordinary forwards, CPU state gates."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from specrhythm.phase4.batched_draft_service import (
    BatchedDraftStateMachine,
    write_immutable_report,
)
from specrhythm.phase4.draft_admission import admit_device, collect_runtime
from specrhythm.phase4.draft_d3_diagnostics import assert_private_blocks
from specrhythm.phase4.draft_gate import _hf_fixture, _proposal_row
from specrhythm.phase4.draft_logits_contract import load_probe_fixture, require
from specrhythm.phase4.draft_logits_probe import preflight
from specrhythm.phase4.draft_qualification import (
    STRUCTURAL_CHECKS,
    compatible_identity,
    load_probe_qualification,
    qualify_d3,
)
from specrhythm.phase4.manifest import sha256_file
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.phase4.vllm_draft_backend import VllmBatchedDraftBackend


def state_snapshot(backend, machine, *, committed):
    """Read existing host state at API boundaries; no hooks or GPU tensor reads.

    Physical K/V bytes are not inspected. The independent four-path equality
    evidence supplies the qualified execution-history control.
    """
    worker, states = backend.worker, backend.states
    internal = {s.internal_id: rid for rid, s in states.items()}
    require(len(internal) == len(states), "duplicate internal request identity")
    require(set(worker.views) == set(internal), "allocator/request identity mismatch")
    require(set(worker.runner.requests) == set(internal), "runner/request identity mismatch")
    owners = {rid: [list(g) for g in worker.kv.get_block_ids(rid)] for rid in internal}
    assert_private_blocks(owners)
    rows = []
    for internal_id, rid in internal.items():
        state, logical = states[rid], machine.requests[rid]
        context = tuple(worker.views[internal_id].prompt_token_ids)
        require(state.prefix == logical.committed_token_ids, "committed prefix identity mismatch")
        require(state.next_round == logical.next_round_id, "round identity mismatch")
        require(
            not logical.finished and rid not in backend.retired, "retired request remained live"
        )
        require(state.materialized == len(context), "materialized frontier mismatch")
        require(
            context[: len(state.prefix)] == state.prefix, "physical view has a different prefix"
        )
        require(
            bool(owners[internal_id]) and all(owners[internal_id]), "missing physical KV ownership"
        )
        if committed:
            require(
                state.materialized == len(state.prefix) and state.proposal is None,
                "committed KV length differs",
            )
        cached = worker.runner.requests[internal_id]
        require(tuple(cached.prompt_token_ids) == context, "runner cached context mismatch")
        rows.append(
            {
                "request_id": rid,
                "internal_id": internal_id,
                "round": state.next_round,
                "committed_prefix_sha256": token_prefix_hash(state.prefix),
                "committed_prefix_len": len(state.prefix),
                "materialized_kv_length": state.materialized,
                "physical_block_ids": owners[internal_id],
            }
        )
    batch = worker.runner.input_batch
    require(set(batch.req_ids) <= set(internal), "InputBatch contains an unknown/retired request")
    for index, rid in enumerate(batch.req_ids):
        require(batch.req_id_to_index[rid] == index, "batch row mapping mismatch")
        view = worker.views[rid]
        count = len(view.prompt_token_ids)
        require(int(batch.num_prompt_tokens[index]) == count, "InputBatch context length mismatch")
        require(
            batch.token_ids_cpu[index, :count].tolist() == list(view.prompt_token_ids),
            "InputBatch prefix tokens mismatch",
        )
        require(
            int(batch.num_computed_tokens_cpu[index])
            == worker.runner.requests[rid].num_computed_tokens,
            "CPU materialized frontier mismatch",
        )
        for group, table in enumerate(batch.block_table.block_tables):
            require(table.blocks_per_kv_block == 1, "unaudited split KV block layout")
            n = int(table.num_blocks_per_row[index])
            require(
                table.get_numpy_array()[index, :n].tolist() == owners[rid][group],
                "CPU block table differs from physical owner",
            )
    return {
        "committed_boundary": committed,
        "rows": rows,
        "input_batch_order": list(batch.req_ids),
        "checks_passed": True,
    }


def replay_production(machine, fixture, vocab_size, snapshot, progress=None):
    """Synthetic verification follows actual vLLM proposals, never HF token deltas."""
    backend = machine.backend
    prefixes = {rid: tuple(p) for rid, p in fixture["initial"].items()}
    limits = {rid: 10 + i % 3 for i, rid in enumerate(prefixes)}
    generated = dict.fromkeys(prefixes, 0)
    for rid, prefix in prefixes.items():
        machine.initialize(rid, prefix, token_prefix_hash(prefix))
    snapshots, comparisons, commits, cohort_sizes = [], [], [], []
    if progress is not None:
        progress.update(
            comparisons=comparisons,
            commits=commits,
            state_snapshots=snapshots,
            cohort_sizes=cohort_sizes,
        )
    reference_prefixes = dict(prefixes)
    eos_retired = set()
    for round_id, reference in enumerate(fixture["rounds"]):
        snapshots.append(snapshot(backend, machine, committed=True))
        refs = {row["request_id"]: row for row in reference["proposals"]}
        rows = []
        for rid, prefix in prefixes.items():
            # Natural EOS plus the original recorded subset policy. A changed
            # initial EOS trajectory fails the explicit subset gate below.
            require(rid in refs, "production EOS cohort differs from qualification fixture")
            rows.append(
                _proposal_row(
                    rid, prefix, round_id, limits[rid] - generated[rid], refs[rid]["eos_token_ids"]
                )
            )
        response = machine.batch_propose(rows)
        by_id = {r["request_id"]: r for r in response["proposals"]}
        require(
            len(by_id) == len(rows) and set(by_id) == set(prefixes),
            "proposal request identity mismatch",
        )
        cohort_sizes.append(len(rows))
        snapshots.append(snapshot(backend, machine, committed=False))
        sync, following = [], {}
        for row in rows:
            rid, prefix = row["request_id"], prefixes[row["request_id"]]
            proposal = by_id[rid]
            tokens = tuple(proposal["proposal_token_ids"])
            eos = row["eos_token_ids"]
            require(
                proposal["round_id"] == round_id
                and proposal["parent_prefix_hash"] == token_prefix_hash(prefix)
                and proposal["parent_prefix_len"] == len(prefix),
                "proposal round/parent identity mismatch",
            )
            require(
                0 < len(tokens) <= min(4, row["remaining_output_budget"] - 1),
                "proposal output budget violation",
            )
            require(not any(t in eos for t in tokens[:-1]), "proposal continued after EOS")
            require(proposal["proposal_eos"] is (tokens[-1] in eos), "proposal EOS flag mismatch")
            same = prefix == reference_prefixes.get(rid)
            comparisons.append(
                {
                    "request_id": rid,
                    "round": round_id,
                    "parent_prefix_sha256": token_prefix_hash(prefix),
                    "same_prefix": same,
                    "hf_tokens": reference["expected"].get(rid),
                    "vllm_tokens": list(tokens),
                    "exact": list(tokens) == reference["expected"].get(rid) if same else None,
                }
            )
            terminal = tokens[-1] in eos or round_id == 3
            if tokens[-1] in eos:
                accepted, tail = len(tokens), ()
                eos_retired.add(rid)
            else:
                accepted = 0 if round_id == 0 else 1 if round_id == 1 else len(tokens)
                correction = (tokens[min(accepted, len(tokens) - 1)] + 7) % vocab_size
                while correction in eos:
                    correction = (correction + 1) % vocab_size
                tail = (correction,)
            delta = tokens[:accepted] + tail
            final = prefix + delta
            generated[rid] += len(delta)
            require(generated[rid] <= limits[rid], "committed output exceeds budget")
            if terminal and rid not in eos_retired:
                require(
                    generated[rid] == limits[rid],
                    "length termination did not consume output budget",
                )
            sync.append(
                {
                    "request_id": rid,
                    "round_id": round_id,
                    "committed_delta": list(delta),
                    "committed_prefix_hash": token_prefix_hash(final),
                    "terminal": terminal,
                }
            )
            if not terminal:
                following[rid] = final
        committed = machine.synchronize_and_batch_propose(sync, [])
        actual_sync = committed["synchronizations"]
        require(
            [r["request_id"] for r in actual_sync] == [r["request_id"] for r in sync],
            "commit request mapping mismatch",
        )
        for expected, actual in zip(sync, actual_sync):
            state = actual["state"]
            rid = expected["request_id"]
            length = len(prefixes[rid]) + len(expected["committed_delta"])
            require(
                state["round_id"] == round_id
                and state["committed_prefix_hash"] == expected["committed_prefix_hash"],
                "commit round/hash mismatch",
            )
            require(
                state["materialized_kv_length"]
                == state["logical_draft_kv_length"]
                == state["committed_prefix_len"]
                == length,
                "commit acknowledged wrong KV frontier",
            )
            require(
                state["draft_internal_request_id"] == f"sr-draft:{rid}"
                and state["draft_physical_request_block_observable"] is True,
                "commit physical identity unobserved",
            )
            require(state["finished"] is expected["terminal"], "retirement flag differs")
            if expected["terminal"]:
                require(
                    rid in backend.retired
                    and rid not in backend.states
                    and f"sr-draft:{rid}" not in backend.worker.views,
                    "retired request retained KV",
                )
        commits.extend(actual_sync)
        snapshots.append(snapshot(backend, machine, committed=True))
        reference_prefixes = {
            r["request_id"]: reference_prefixes[r["request_id"]] + tuple(r["committed_delta"])
            for r in reference["synchronizations"]
            if not r["terminal"]
        }
        prefixes = following
    count = len(fixture["initial"])
    require(not prefixes and len(backend.retired) == count, "requests did not all retire")
    require(
        0 < len(eos_retired) < count and any(0 < n < count for n in cohort_sizes),
        "shrinking EOS subset was not executed",
    )
    return {
        "comparisons": comparisons,
        "commits": commits,
        "state_snapshots": snapshots,
        "cohort_sizes": cohort_sizes,
        "eos_retired_requests": sorted(eos_retired),
        "completed_requests": count,
        "generated_tokens_by_request": generated,
        "output_limits_by_request": limits,
    }


def run_qualification(config, count, output, identity, regime, *, preflight_error=None):
    output.mkdir(parents=True, exist_ok=False)
    write_immutable_report(output / "regime-qualification.json", regime)
    observation = {
        "requested_batch_size": count,
        "run_identity": identity,
        "errors": [],
        "diagnostic_only": False,
        "materialization_observer_installed": False,
        "execution_started": False,
        "admission_valid": False,
        "structural_checks_executed": False,
        "structural_checks": dict.fromkeys(STRUCTURAL_CHECKS, None),
        "admission": {
            "schema_version": "specrhythm.phase4b3-draft-admission.v1",
            "valid": False,
            "errors": [],
            "historical_probe_gpu_uuid": regime.get("gpu_identity", {}).get("gpu_uuid"),
            **dict.fromkeys(
                (
                    "current_device_binding_valid",
                    "current_gpu_uuid",
                    "same_physical_gpu",
                    "execution_regime_device_compatible",
                    "execution_regime_compatible",
                    "current_runtime_provenance",
                ),
                None,
            ),
        },
        "comparisons": [],
    }
    backend = None
    evidence = {}
    try:
        require(preflight_error is None, preflight_error or "preflight failed")
        require(
            regime["valid"] and compatible_identity(identity, regime["run_identity"]),
            "missing/incompatible qualified execution regime",
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(config.draft.resolved_tokenizer_path),
            revision=config.draft.tokenizer_revision,
            trust_remote_code=config.draft.trust_remote_code,
        )
        fixtures = _hf_fixture(config, [(f"batch-{count}", count, True)], tokenizer)
        write_immutable_report(output / "hf-oracle.json", {"fixtures": fixtures})
        # Original HF fixture generation finishes/destroys its model first.
        backend = VllmBatchedDraftBackend(config)
        write_immutable_report(output / "draft-startup.json", backend.provenance)
        runtime = collect_runtime(backend)
        observation["runtime_batch_invariance"] = runtime["batch_invariance"]
        observation["admission"] = admit_device(runtime, identity, regime)
        observation["admission_valid"] = observation["admission"]["valid"]
        require(
            observation["admission_valid"],
            "D3 admission failed: " + "; ".join(observation["admission"]["errors"]),
        )
        machine = BatchedDraftStateMachine(backend)
        vocab = json.loads((config.draft.resolved_model_path / "config.json").read_text())[
            "vocab_size"
        ]
        observation["execution_started"] = True
        observation.update(
            replay_production(machine, fixtures[0], vocab, state_snapshot, observation)
        )
        observation["structural_checks"] = dict.fromkeys(STRUCTURAL_CHECKS, True)
        observation["structural_checks_executed"] = True
        machine.shutdown()
    except Exception as error:
        observation["errors"].append(f"{type(error).__name__}: {error}")
        observation["traceback"] = traceback.format_exc()
        if not observation["admission"]["errors"] and not observation["admission_valid"]:
            observation["admission"]["errors"] = list(observation["errors"])
    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as error:
                observation["errors"].append(f"cleanup: {error}")
                observation["structural_checks"]["cleanup"] = False
            evidence = backend.report()
        write_immutable_report(output / "observation.json", observation)
        write_immutable_report(output / "draft-backend-report.json", evidence)
        write_immutable_report(output / "admission.json", observation["admission"])
    result = qualify_d3(observation, evidence, regime)
    result["artifact_sha256"] = {p.name: sha256_file(p) for p in output.glob("*.json")}
    write_immutable_report(output / "gate.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--request-count", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("D3 requires operator --allow-gpu")
    regime, identity, config, error = {}, {}, None, None
    try:
        regime = load_probe_qualification(args.probe_root)
        config, identity = preflight(args.config, args.expected_commit, load_probe_fixture())
    except Exception as exc:
        error = f"preflight: {type(exc).__name__}: {exc}"
    result = run_qualification(
        config, args.request_count, args.output, identity, regime, preflight_error=error
    )
    print(json.dumps({k: result[k] for k in ("d3_qualified", "hf_draft_exact", "errors")}))
    return 0 if result["d3_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
