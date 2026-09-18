"""Ordinary K3 double-buffer execution commands; no second engine or KV owner.

The owner still claims at a fenced token boundary. A command is immutable until
the single coordinator collects engine.step(); the other home runs on the existing
Draft owner throughout. Only model provenance is interned within a wire message.
Live prefix/version, tokens, ownership and materialization are never cached.
"""

import functools
import os
from contextlib import contextmanager
from contextvars import ContextVar

from specrhythm.serving.common import require

SCHEMA = "specrhythm.dual-batch-command.v1"
_VIEW = ContextVar("dual_batch_control", default=None)


def enabled():
    return os.environ.get("SR_K3_TARGET_DISPATCH") == "dual-batch"


def validate_mode(mode):
    require(mode == "pingpong-k3", "dual-batch requires ordinary pingpong-k3")


def pack(admission):
    """Lossless call-local interning, including heterogeneous provenance.

    All request-specific fields survive. Equality is of contents, not object ID;
    at most one entry per claimed request, bounded by the real Target ceiling.
    """
    models, claims = [], []
    require(len(admission["claims"]) <= 128, "unbounded dual-batch claims")
    for claim in admission["claims"]:
        proposal = claim["proposal"]
        model = proposal["model_provenance"]
        require(isinstance(model, dict), "invalid proposal model provenance")
        index = next((i for i, m in enumerate(models) if m == model), None)
        if index is None:
            index = len(models)
            models.append(model)
        claims.append({**claim, "proposal": {
            **{k: v for k, v in proposal.items() if k != "model_provenance"},
            "model_reference": index,
        }})
    return {"schema_version": SCHEMA, "models": models,
            "admission": {**admission, "claims": claims}}


def unpack(command):
    require(command.get("schema_version") == SCHEMA, "invalid dual-batch wire schema")
    models, admission = command["models"], command["admission"]
    require(isinstance(models, list) and len(models) <= 128
            and all(isinstance(m, dict) for m in models), "invalid model table")
    require(len(admission["claims"]) <= 128, "unbounded dual-batch claims")
    claims = []
    for claim in admission["claims"]:
        proposal = claim["proposal"]
        index = proposal["model_reference"]
        require(type(index) is int and 0 <= index < len(models)
                and "model_provenance" not in proposal, "invalid model reference")
        claims.append({**claim, "proposal": {
            **{k: v for k, v in proposal.items() if k != "model_reference"},
            "model_provenance": models[index],
        }})
    return {**admission, "claims": claims}


def encode_control(value):
    require("dual_batch_command" not in value, "command is already encoded")
    if "pp_admission" not in value:
        return value  # Resident setup/drain without a dispatched proposal.
    return {**{k: v for k, v in value.items() if k != "pp_admission"},
            "dual_batch_command": pack(value["pp_admission"])}


def decode_control(value):
    if "dual_batch_command" not in value:
        return value
    require(enabled() and "pp_admission" not in value, "unexpected dual-batch control")
    validate_mode(os.environ.get("SR_S2_MODE"))
    return {**{k: v for k, v in value.items() if k != "dual_batch_command"},
            "pp_admission": unpack(value["dual_batch_command"])}


def current_control():
    return _VIEW.get() if enabled() else None


@contextmanager
def control_scope():
    """One synchronous schedule/verify call, never across engine calls or threads.

    The coordinator cannot publish a new command during its synchronous Target
    call. Always re-read on entry; nested pool/proposer consumers share that same
    command. Even exceptional exits invalidate it. This is not a KV/state cache.
    """
    if not enabled() or _VIEW.get() is not None:
        yield
        return
    from specrhythm.serving.s2_pool import control

    packet = control()
    token = _VIEW.set(packet)
    try:
        yield
    finally:
        _VIEW.reset(token)


def control_transaction(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with control_scope():
            return function(*args, **kwargs)
    return call


class CommandPublisher:
    """Coordinator-owned snapshots; no recursive copies of claimed proposals.

    Clock.control creates fresh scalar lifecycle rows; packet mappings are copied
    on publication. Admission is a newly received, read-only response, replaced
    wholesale at the next opportunity. No generated prefix or KV is retained.
    """
    def __init__(self, path):
        self.path, self.previous, self.admission = path, None, None

    def publish(self, current):
        from specrhythm.serving.s2_pool import publish

        admission = current.get("pp_admission")
        base = {k: v for k, v in current.items() if k != "pp_admission"}
        for key in ("initial_proposals", "initial_enqueues"):
            if key in base:
                # K3 does not store proposals here. Refuse accidental reuse of
                # the legacy mutable initial-proposal protocol in this path.
                require(not base[key], "dual-batch unexpected legacy initial proposals")
                base[key] = {}
        if base != self.previous or admission is not self.admission:
            publish(self.path, current)
            self.previous, self.admission = base, admission


def run_target(engine):
    """Submit and collect on the sole engine caller; Draft owner is independent.

    No future, extra thread, new barrier or wait for other-home READY. Worker
    feedback is already acknowledged before this returns, while settlement can
    still be pending on the Draft owner. Preserve that distinction.
    """
    validate_mode(os.environ.get("SR_S2_MODE"))
    with control_scope():
        return engine.step()
