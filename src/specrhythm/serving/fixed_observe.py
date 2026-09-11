"""Opt-in in-memory diagnostic timers. No per-forward writes or new per-round fences."""

from __future__ import annotations

import functools
import os
import time
from collections import defaultdict
from contextlib import contextmanager

from specrhythm.serving.common import require


class Timers:
    def __init__(self):
        self.rows = []

    @contextmanager
    def span(self, category, **fields):
        start = time.monotonic_ns()
        try:
            yield
        finally:
            self.rows.append(
                dict(category=category, start_ns=start, end_ns=time.monotonic_ns(), **fields)
            )

    def report(self):
        counts = defaultdict(lambda: {"count": 0, "inclusive_host_ms": 0.0})
        for r in self.rows:
            counts[r["category"]]["count"] += 1
            counts[r["category"]]["inclusive_host_ms"] += (r["end_ns"] - r["start_ns"]) / 1e6
        return {
            "aggregate": dict(counts),
            "intervals": list(self.rows),
            "aggregation": "inclusive/nested categories; do not add or infer critical path",
        }


TIMERS = Timers()
ROUNDS = []
TARGET_ROWS = []
_INSTALLED = False


def wrap(owner, name, category):
    original = getattr(owner, name)
    if getattr(original, "_fixed_observer", False):
        return original

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with TIMERS.span(category):
            return original(*args, **kwargs)

    measured._fixed_observer = True
    setattr(owner, name, measured)
    return measured


def install_host_observation(role=None):
    """Called only by the independent diagnostic child/worker startup, never S1/S2."""
    global _INSTALLED
    if _INSTALLED:
        return
    require(bool(os.environ.get("SR_FIXED_POINT")), "diagnostic observation requires a point")
    _INSTALLED = True
    import json
    import subprocess

    from specrhythm.phase4 import dual_service, transport
    from specrhythm.phase4.dual_uuid import DualVerificationUuidQuery
    from specrhythm.phase4.vllm_draft_worker import VllmDraftWorker
    from specrhythm.serving import s2_pool

    wrap(DualVerificationUuidQuery, "for_verification", "live_uuid_validation")
    from specrhythm.serving.fixed_logging import install_checkpoint_logging

    install_checkpoint_logging(role, capture, TIMERS)
    wrap(os, "fsync", "log_fsync")
    wrap(transport.UnixDraftClient, "call", "ipc")
    wrap(dual_service.DualDraftClient, "call", "ipc")
    wrap(s2_pool.ResidentPoolAudit, "check", "resident_block_audit")
    wrap(VllmDraftWorker, "fence", "draft_required_fence")
    # Wrap aliases in their defining modules before consumers import them.
    for name in ("control", "publish", "prefix_record"):
        original_pool_function = getattr(s2_pool, name)
        measured = wrap(
            s2_pool,
            name,
            {
                "control": "control_json_read",
                "publish": "control_json_write",
                "prefix_record": "prefix_hash_and_block_record",
            }[name],
        )
        import sys

        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("specrhythm.") and module is not None:
                for key, value in list(vars(module).items()):
                    if value is original_pool_function:
                        setattr(module, key, measured)
    for name in ("dump", "dumps", "load", "loads"):
        wrap(json, name, "json_serialization")
    original = subprocess.run

    @functools.wraps(original)
    def query(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and command and "nvidia-smi" in str(command[0]):
            with TIMERS.span("nvidia_smi_subprocess"):
                return original(command, *args, **kwargs)
        return original(command, *args, **kwargs)

    subprocess.run = query
    # GPU imports occur here only in explicit server children/worker RPCs.
    import torch
    from vllm.distributed.parallel_state import GroupCoordinator

    wrap(torch.cuda, "synchronize", "required_cuda_synchronize")
    wrap(torch.cuda.Event, "synchronize", "required_cuda_event_synchronize")
    wrap(GroupCoordinator, "barrier", "required_tp_barrier")


class DeviceTimeline:
    """CUDA event intervals with bracketed common-monotonic anchor uncertainty.

    Startup anchor synchronization is outside the measured window. Events are read
    after the existing final fence. Cross-device overlap is reported as a bound,
    not exact kernel overlap; TP durations are retained separately and never summed.
    """

    def __init__(self, torch, model, metadata, *, identity):
        self.torch, self.metadata, self.identity = torch, metadata, identity
        self.pending = []
        self.current = None
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor_before_ns = time.monotonic_ns()
        self.anchor.record()
        self.anchor.synchronize()
        self.anchor_after_ns = time.monotonic_ns()
        self.hooks = [
            model.register_forward_pre_hook(self.before),
            model.register_forward_hook(self.after),
        ]

    def before(self, _module, _args):
        start = self.torch.cuda.Event(enable_timing=True)
        meta = self.metadata()
        host = time.monotonic_ns()
        start.record()
        self.current = (start, host, meta)

    def after(self, _module, _args, _output):
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        require(self.current is not None, "GPU forward end without start")
        start, host, meta = self.current
        self.pending.append((start, end, host, time.monotonic_ns(), meta))
        self.current = None

    def report(self):
        from specrhythm.serving.fixed_timing import project_anchor

        rows = []
        for start, end, host, launch_end, meta in self.pending:
            require(end.query(), "final fence did not complete diagnostic CUDA events")
            a, b = self.anchor.elapsed_time(start) * 1e6, self.anchor.elapsed_time(end) * 1e6
            require(b > a, "nonpositive diagnostic GPU event duration")
            rows.append(
                {
                    **meta,
                    "host_start_ns": host,
                    "host_launch_end_ns": launch_end,
                    "gpu_event_ms": (b - a) / 1e6,
                    **project_anchor(self.anchor_before_ns, self.anchor_after_ns, a, b),
                }
            )
        return {
            "identity": self.identity,
            "forwards": rows,
            "clock": "CUDA elapsed events projected onto bracketed host monotonic anchor",
            "projection_version": "integer-anchor-v2",
            "anchor_before_ns": self.anchor_before_ns,
            "anchor_after_ns": self.anchor_after_ns,
            "anchor_uncertainty_ns": self.anchor_after_ns - self.anchor_before_ns,
            "extra_per_round_synchronization": False,
            "kernel_overlap_exact": False,
        }


def target_startup(worker):
    install_host_observation()
    from specrhythm.serving.s2_runtime import initialize_pingpong_worker, target_snapshot

    snapshot = (
        initialize_pingpong_worker if os.environ["SR_S2_MODE"] == "pingpong" else target_snapshot
    )(worker)
    from specrhythm.serving.fixed_logging import current

    current("target-rank-" + str(snapshot["global_rank"]))
    runner = worker.model_runner
    import torch

    def metadata():
        ids = list(runner.input_batch.req_ids)
        return {"internal_request_ids": ids, "B": len(ids), "role": "target"}

    worker.fixed_timeline = DeviceTimeline(
        torch,
        worker.get_model(),
        metadata,
        identity={
            **{
                k: snapshot[k]
                for k in ("global_rank", "local_rank", "gpu_uuid", "physical_gpu_id")
            },
            "role": "target",
            "dtype": str(worker.vllm_config.model_config.dtype),
            "attention_backends": snapshot["attention_backends"],
        },
    )
    return snapshot


def target_report(worker):
    from specrhythm.serving.fixed_logging import current

    logs = current()
    # The coordinator has already performed its normal final target_fence RPC.
    return {
        "device": worker.fixed_timeline.report(),
        "diagnostic_logging": logs.snapshot() if logs else None,
        "host": TIMERS.report(),
        "rounds": ROUNDS,
        "target_rows": TARGET_ROWS,
    }


def capture(row):
    """Retain compact producer evidence in memory, never reread giant logs for the light path."""
    schema = row.get("schema_version")
    if schema in ("specrhythm.phase4-round-event.v1", "specrhythm.phase4b-proposal-event.v1"):
        if "committed_token_ids" in row:
            ROUNDS.append(
                {
                    k: v
                    for k, v in row.items()
                    if k
                    in (
                        "request_id",
                        "round_id",
                        "proposal_token_ids",
                        "accepted_draft_token_ids",
                        "rejected_draft_token_ids",
                        "target_correction_token_ids",
                        "target_bonus_token_ids",
                        "committed_token_ids",
                        "accepted_draft_tokens",
                        "rejected_draft_tokens",
                        "parent_prefix_len",
                        "prefix_token_count",
                        "prefix_version",
                        "timeline",
                        "commit_start_ns",
                        "commit_end_ns",
                        "terminal",
                        "verify_microbatch_id",
                    )
                }
            )
    if schema == "specrhythm.phase4-target-forward-diagnostic.v1":
        from specrhythm.phase4.vllm_diagnostics import validate_target_diagnostic

        with TIMERS.span("target_diagnostic_contract_scan"):
            errors = validate_target_diagnostic(row)
        TARGET_ROWS.append(
            {
                k: row.get(k)
                for k in (
                    "request_id",
                    "target_forward_start_ns",
                    "target_forward_end_ns",
                    "query_length",
                    "position_ids",
                    "proposal_token_ids",
                    "target_input_token_ids",
                    "dtype",
                    "attention_backend",
                    "batch_invariant_requested",
                )
            }
            | {
                "context_length": len(row["committed_prefix_token_ids"]),
                "structural_errors": errors,
            }
        )
