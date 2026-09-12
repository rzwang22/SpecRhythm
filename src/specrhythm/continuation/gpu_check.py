"""Operator-only physical Draft KV regression; inspection never initializes CUDA.

Controlled receipt constructions exercise repair, not observed Target outcomes.
Actual Target correctness/overlap/performance belongs to the serial-eager run.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from specrhythm.continuation.core import DraftCompletion, ParentVerification, RollingContinuation
from specrhythm.continuation.policy import StaticEagerEligibility
from specrhythm.phase4.draft_batch import DraftCommitPlan, DraftProposalPlan
from specrhythm.phase4.dual_commit import dual_greedy_acceptance
from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import read_json, require
from specrhythm.serving.s2_pool import publish

DIRECTORY = "rolling-eager-gpu-check"
CASES = (
    "success", "success", "parent_rejected", "success", "success", "bridge_mismatch", "success"
)


def physical_snapshot(backend):
    """Reuse the qualified host metadata/page-table audit on the actual worker."""
    from specrhythm.phase4.draft_qualification_gate import state_snapshot

    logical = SimpleNamespace(requests={
        rid: SimpleNamespace(
            committed_token_ids=s.prefix, next_round_id=s.next_round, finished=False
        ) for rid, s in backend.states.items()
    })
    return state_snapshot(backend, logical, committed=False)


def _alternative(token, eos, vocab_size):
    for step in range(1, vocab_size):
        candidate = (token + step) % vocab_size
        if candidate not in eos:
            return candidate
    raise ValueError("no non-EOS correction token in vocabulary")


def run_checks(backend, requests, *, eos_token_ids, vocab_size, snapshot=physical_snapshot,
               checkpoint=lambda event: None):
    """Run the real backend/shared-core protocol; injectable worker operations are for CI.

    Only independent diagnostic reference requests replay full prefixes. The live
    rolling request initializes once, then keeps/repairs its existing private KV.
    """
    ids = tuple(r["request_id"] for r in requests)
    require(ids and len(ids) == len(set(ids)), "GPU check requires unique requests")
    core = RollingContinuation("gpu-correctness-owner", StaticEagerEligibility(frozenset(ids)))
    events, comparisons, reference_prefills = [], [], [0]
    observed = dict.fromkeys(CASES, 0)

    def record(event):
        events.append(event)
        checkpoint(event)

    def normal_many(request_ids):
        works = {rid: core.schedule_normal_recovery(rid) for rid in request_ids}
        tokens = backend.propose_many(tuple(DraftProposalPlan(
            rid, backend.states[rid].next_round, work.dependency_prefix,
            work.candidate_length, eos_token_ids) for rid, work in works.items()))
        return {rid: core.record_normal_completion(work, DraftCompletion(
            work.work_id, tokens[rid], backend.states[rid].materialized))
            for rid, work in works.items()}

    def compare(proposal, stage):
        ref = f"gpu-check-reference:{proposal.request_id}:{stage}"
        backend.initialize(ref, proposal.parent_prefix)
        reference_prefills[0] += 1
        try:
            expected = backend.propose_many((DraftProposalPlan(
                ref, 0, proposal.parent_prefix, len(proposal.tokens), eos_token_ids
            ),))[ref]
            comparison = {
                "request_id": proposal.request_id, "stage": stage,
                "prefix_sha256": token_prefix_hash(proposal.parent_prefix),
                "proposal_tokens": list(proposal.tokens), "reference_tokens": list(expected),
                "exact": proposal.tokens == expected,
                "reference": "ordinary same-worker Draft on independent private KV",
            }
            comparisons.append(comparison)
            require(comparison["exact"], "incremental GPU Draft differs from ordinary reference",
                    request_id=proposal.request_id, stage=stage)
            snapshot(backend)
        finally:
            backend.finish_many((ref,))

    for row in requests:
        rid, prefix = row["request_id"], tuple(row["prefix"])
        core.register(rid, prefix, max_output_tokens=row["remaining_output_tokens"],
                      eos_token_ids=eos_token_ids)
        backend.initialize(rid, prefix)
    record({"event": "initialized", "physical": snapshot(backend)})
    proposals = normal_many(ids)
    for stage in range(len(CASES)):
        if not proposals:
            break
        works, currents = {}, {}
        for rid, proposal in proposals.items():
            compare(proposal, stage)
            currents[rid] = core.state(rid)
            core.start_verification(rid, proposal.proposal_id)
            work = core.begin_continuation(rid)
            if work is not None:
                works[rid] = work
        backend.begin_gpu_continuations(tuple(works.values()))
        completions = backend.step_gpu_continuations(tuple(works.values())) if works else {}
        pending = []
        for rid, proposal in proposals.items():
            # Offset the same seven-case cycle to exercise mixed parent outcomes
            # in one physical batch, while retaining each request's case counts.
            construction = CASES[(stage + 3*ids.index(rid)) % len(CASES)]
            current, work = currents[rid], works.get(rid)
            bridge = (backend.gpu_continuation_snapshot(work)["generated_tokens"][0]
                      if work is not None else None)
            terminal = None
            if proposal.tokens and proposal.tokens[-1] in eos_token_ids:
                delta, terminal = proposal.tokens, "eos"
            elif not proposal.tokens:
                value = backend.worker.greedy((backend.states[rid].next_logits,))[0]
                delta = (value,)
                terminal = "eos" if value in eos_token_ids else "length"
            elif work is None:
                # A separate diagnostic reference obtains the actual bonus while
                # preserving the live request's strict ordinary proposal frontier.
                ref = f"gpu-check-bonus-reference:{rid}:{stage}"
                backend.initialize(ref, proposal.parent_prefix + proposal.tokens)
                reference_prefills[0] += 1
                try:
                    bridge = backend.worker.greedy((backend.states[ref].next_logits,))[0]
                finally:
                    backend.finish_many((ref,))
                delta = proposal.tokens + (bridge,)
                terminal = "eos" if bridge in eos_token_ids else None
            elif construction == "parent_rejected":
                accepted = min(2, len(proposal.tokens) - 1)
                delta = proposal.tokens[:accepted] + (
                    _alternative(proposal.tokens[accepted], eos_token_ids, vocab_size),
                )
            elif construction == "bridge_mismatch":
                delta = proposal.tokens + (_alternative(bridge, eos_token_ids, vocab_size),)
            else:
                delta = proposal.tokens + (bridge,)
                if bridge in eos_token_ids:
                    terminal = "eos"
            if len(delta) == current.remaining_output_tokens and terminal is None:
                terminal = "length"
            receipt = ParentVerification(
                core.owner_id, rid, proposal.proposal_id, proposal.prefix_version,
                proposal.parent_prefix, delta, terminal,
            )
            decision = dual_greedy_acceptance(
                proposal.tokens, delta, terminal=terminal is not None
            )
            plan = DraftCommitPlan(
                rid, backend.states[rid].next_round, proposal.parent_prefix, proposal.tokens,
                len(decision.accepted_draft_token_ids),
                decision.target_correction_token_ids + decision.target_bonus_token_ids,
                token_prefix_hash(proposal.parent_prefix + delta), terminal is not None,
                len(decision.target_correction_token_ids), len(decision.target_bonus_token_ids),
            )
            pending.append((rid, construction, receipt, plan, work, terminal))
        if stage % 2:
            for _, _, receipt, _, _, _ in pending:
                core.resolve_parent_verification(receipt)
        for _ in range(4):
            if all(c is not None for c in completions.values()):
                break
            completions = backend.step_gpu_continuations(tuple(works.values()))
        require(all(c is not None for c in completions.values()),
                "bounded K4 GPU continuation did not finish")
        for work in works.values():
            core.record_continuation_completion(work, completions[work.work_id])
        if not stage % 2:
            for _, _, receipt, _, _, _ in pending:
                core.resolve_parent_verification(receipt)
        next_proposals, settlements = {}, []
        for rid, _, _, plan, work, terminal in pending:
            promoted = None
            if not terminal and work is not None and (
                core.continuation(rid, work.work_id).status == "promotable"
            ):
                promoted = core.promote_continuation(rid, work.work_id)
                next_proposals[rid] = promoted
            settlements.append((plan, work, promoted.tokens if promoted else None))
        physical_results = backend.rebase_gpu_parents(settlements)
        recovery = []
        for rid, construction, _, _plan, work, terminal in pending:
            observed[construction] += int(terminal is None and work is not None)
            record({
                "event": "controlled_parent_settlement", "request_id": rid, "stage": stage,
                "construction": construction, "receipt_source": "INJECTED_DIAGNOSTIC",
                "target_observation": False, "target_first": bool(stage % 2),
                "terminal_reason": terminal, "promoted": rid in next_proposals,
                "accounting": asdict(core.state(rid).accounting),
                "physical": physical_results[rid], "snapshot": snapshot(backend),
                "settlement_batch_request_ids": [p.request_id for p, _, _ in settlements],
            })
            if not terminal and rid not in next_proposals:
                recovery.append(rid)
        next_proposals.update(normal_many(recovery))
        proposals = next_proposals
    for rid in ids:
        core.finish_or_cancel(rid)
    backend.finish_many(ids)
    core.shutdown()
    require(not backend.states, "GPU correctness left live request KV")
    return {
        "events": events, "reference_comparisons": comparisons,
        "constructed_cases": {
            name: {"count": count, "status": "EXECUTED" if count else "NOT_OBSERVED"}
            for name, count in observed.items()
        },
        "requests": {rid: asdict(core.state(rid).accounting) for rid in ids},
        "live_request_initial_prefills": len(ids),
        "diagnostic_reference_prefills": reference_prefills[0],
        "target_observed_events": "NOT_OBSERVED: no Target model in this Draft-only check",
        "gpu_overlap": "UNKNOWN: no Target model in this Draft-only check",
        "performance_result": False,
    }


def _directory(root):
    return Path(root) / DIRECTORY


def status(root):
    directory = _directory(root)
    return {name: read_json(directory / f"{name}.json")
            if (directory / f"{name}.json").exists() else None
            for name in ("state", "result")}


def errors(root):
    directory = _directory(root)
    return {name: read_json(directory / name) if (directory / name).exists() else None
            for name in ("diagnostic-primary-error.json", "diagnostic-secondary-errors.json")}


def stop(root):
    directory = _directory(root)
    state = status(root)["state"]
    require(state is not None, "GPU correctness check has not started")
    if state["status"] != "RUNNING":
        return {"action": "already stopped", "state": state["status"]}
    publish(directory / "stop-request.json", {"reason": "operator_stop"})
    return {"action": "controlled stop requested", "directory": str(directory)}


def _worker(root, count):
    from specrhythm.continuation.gpu_backend import RollingVllmDraftBackend
    from specrhythm.phase4.config import load_phase4_config
    from specrhythm.serving.fixed_artifacts import record_error
    from specrhythm.serving.schema import load_requests

    directory, backend, value, failure = _directory(root), None, {}, None
    config = load_phase4_config(str(root / "config.json"))
    model = read_json(config.draft.resolved_model_path / "config.json")
    eos = model.get("eos_token_id", ())
    eos = (eos,) if isinstance(eos, int) else tuple(eos or ())
    generation = config.draft.resolved_model_path / "generation_config.json"
    if generation.exists():
        extra = read_json(generation).get("eos_token_id", ())
        eos = tuple(dict.fromkeys((*eos, *((extra,) if isinstance(extra, int) else extra or ()))))
    rows = load_requests(root / "inputs/requests.jsonl")[:count]
    requests = [{"request_id": f"gpu-check:{row.request_id}", "prefix": row.prompt_token_ids,
                 "remaining_output_tokens": min(row.maximum_new_tokens, 64)} for row in rows]

    def interrupted(signum, frame):
        raise RuntimeError("GPU correctness worker received controlled stop")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        backend = RollingVllmDraftBackend(config)
        require(backend.provenance.get("physical_gpu_id") == 0,
                "GPU correctness must use the real Draft device")
        with (directory / "events.jsonl").open("x") as events:
            def save(event):
                events.write(json.dumps(event, sort_keys=True) + "\n")
                events.flush()

            value = run_checks(backend, requests, eos_token_ids=eos,
                               vocab_size=model["vocab_size"], checkpoint=save)
    except BaseException as error:
        failure = error
        record_error(directory, error, "gpu_correctness")
    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except BaseException as error:
                record_error(directory, error, "gpu_correctness_shutdown")
                failure = failure or error
            value["backend"] = backend.report()
        resources = value.get("backend", {}).get("worker_resources", {})
        cleanup = (backend is not None and not backend.states and backend.closed
                   and resources.get("live_allocator_requests") == 0
                   and resources.get("blocks_allocated") == resources.get("blocks_freed")
                   and resources.get("worker_shutdown_complete") is True)
        value.update(schema_version="specrhythm.rolling-eager-gpu-check.v1",
                     valid=failure is None and cleanup,
                     cleanup_status="PASS" if cleanup else "FAILED",
                     physical_gpu_correctness="PASS" if failure is None and cleanup else "FAILED",
                     error=None if failure is None else str(failure), performance_result=False)
        publish(directory / "result.json", value)
    return 0 if value["valid"] else 1


def supervise(directory, command, env, *, timeout, drain_timeout):
    """Own a single worker tree; stop requests and failures share one drain deadline."""
    from specrhythm.phase4.owned_processes import set_subreaper

    previous = set_subreaper(True)
    try:
        return _supervise(directory, command, env, timeout=timeout, drain_timeout=drain_timeout)
    finally:
        if previous is not None:
            set_subreaper(bool(previous))


def _supervise(directory, command, env, *, timeout, drain_timeout):
    from specrhythm.phase4.owned_processes import OwnedProcesses, process_table
    from specrhythm.serving.fixed_artifacts import record_error

    actions, reaped, failure, child, owner = [], [], None, None, None
    ownership_token = uuid.uuid4().hex
    with (directory / "worker.log").open("x") as log:
        try:
            process_table()  # Prove ownership inspection is available before launching anything.
            child = subprocess.Popen(
                command, env={**env, "SR_PHASE4_OWNED_TARGET_TOKEN": ownership_token},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            owner = OwnedProcesses(child.pid, target_token=ownership_token)
            identity = owner.snapshot()
            publish(directory / "state.json", {"status": "RUNNING", "pid": child.pid,
                                               "owner_identity": identity, "command": command})
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                owner.snapshot()
                stopped = (directory / "stop-request.json").exists()
                if stopped or time.monotonic() >= deadline:
                    raise RuntimeError("operator stop" if stopped else
                                       "GPU correctness execution deadline expired")
                time.sleep(0.1)
            if child.returncode:
                raise RuntimeError(f"GPU correctness worker exited with {child.returncode}")
        except BaseException as error:
            failure = error
            record_error(directory, error, "gpu_correctness_supervisor")
        finally:
            residual = []
            ownership_verified = True
            if owner is not None:
                drain_deadline = time.monotonic() + drain_timeout
                try:
                    if failure is not None:
                        owner.signal(signal.SIGTERM, actions)
                    while owner.snapshot() and time.monotonic() < drain_deadline:
                        child.poll()
                        reaped.extend(owner.reap())
                        time.sleep(0.1)
                    residual = owner.snapshot()
                    if residual:
                        error = RuntimeError("GPU correctness owned cleanup deadline expired")
                        failure = failure or error
                        record_error(directory, error, "gpu_correctness_cleanup")
                        owner.signal(signal.SIGKILL, actions)
                    # Forced cleanup after the deadline cannot qualify as a PASS.
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired as error:
                        failure = failure or error
                        record_error(directory, error, "gpu_correctness_forced_cleanup")
                    reaped.extend(owner.reap())
                    residual = owner.snapshot()
                    if failure is None and any(r["exit_code"] != 0 for r in reaped):
                        failure = RuntimeError("GPU correctness descendant exited unsuccessfully")
                        record_error(directory, failure, "gpu_correctness_descendant")
                except BaseException as error:
                    # Inspection failure must not overwrite the original error or
                    # leave the state falsely RUNNING. Only Popen's direct child
                    # remains safe to signal without a fresh descendant identity.
                    ownership_verified = False
                    failure = failure or error
                    record_error(directory, error, "gpu_correctness_ownership_inspection")
                    residual = list(owner.observed.values())
                    try:
                        child.terminate()
                        child.wait(timeout=max(0.01, drain_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
            result_path = directory / "result.json"
            result = read_json(result_path) if result_path.exists() else {}
            valid = (failure is None and child is not None and child.returncode == 0
                     and ownership_verified and not residual and result.get("valid") is True)
            if not valid and failure is None:
                failure = RuntimeError("GPU correctness worker has no valid result")
                record_error(directory, failure, "gpu_correctness_result")
            publish(directory / "state.json", {
                "status": "COMPLETE" if valid else "FAILED",
                "exit_code": child.returncode if child is not None else None,
                "cleanup_status": (
                    "PASS" if child is not None and ownership_verified and not residual
                    else "FAILED"
                ),
                "ownership_verified": ownership_verified,
                "remaining_owned_pids": [r["pid"] for r in residual], "actions": actions,
                "reaped_descendants": reaped,
                "owner_identity": list(owner.observed.values()) if owner is not None else [],
                "error": None if failure is None else str(failure),
            })
    require(valid, "GPU correctness check failed; retain this root and worker.log")


def run(root, *, count=2, timeout=900, drain_timeout=60):
    """Launch only when explicitly requested; read-only commands never reach this path."""
    from specrhythm.serving.s1_preflight import clean_environment, git_identity
    from specrhythm.serving.s2_cli import root_lock

    require(1 <= count <= 4 and timeout > 0 and 0 < drain_timeout <= 60,
            "invalid bounded GPU correctness settings")
    config = read_json(root / "scan-config.json")
    require(git_identity() == config["execution"]["git_commit"], "GPU check commit differs")
    directory = _directory(root)
    with root_lock(root):
        require(not directory.exists(), "GPU correctness directory must be fresh")
        directory.mkdir()
        env = clean_environment("serial")
        env.update(CUDA_VISIBLE_DEVICES="0", SR_VLLM_SOURCE=config["execution"]["vllm_source"])
        command = [sys.executable, "-m", "specrhythm.continuation.gpu_check", "_worker",
                   "--root", str(root), "--request-count", str(count)]
        supervise(directory, command, env, timeout=timeout, drain_timeout=drain_timeout)
    return status(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "errors", "stop", "_worker"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--drain-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "_worker":
        return _worker(root, args.request_count)
    try:
        value = (run(root, count=args.request_count, timeout=args.timeout,
                     drain_timeout=args.drain_timeout) if args.command == "run"
                 else {"status": status, "errors": errors, "stop": stop}[args.command](root))
        print(json.dumps(value, indent=2, default=str), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"GPU correctness: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
