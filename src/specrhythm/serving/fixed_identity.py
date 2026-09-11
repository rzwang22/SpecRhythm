"""Opt-in reuse of proven frozen prompt bindings; never cache live request state."""

from __future__ import annotations

import os
import threading
import time
from types import MappingProxyType

from specrhythm.phase4.request_identity import FrozenPromptIdentityMap
from specrhythm.serving.common import require

MODES = ("linear", "bound-prefix")
COUNTERS = (
    "bind_calls", "validated_binding_reuses", "full_scans", "candidate_comparisons",
    "linear_candidate_comparisons", "binding_errors", "inclusive_host_ns",
)


class BoundPromptIdentityMap(FrozenPromptIdentityMap):
    """One map per scheduler/proposer; immutable proof, original binding checks.

    Distinct prefix-free prompts cannot both prefix the same physical row. Once
    a binding exists, comparing its ENTIRE frozen prompt with the CURRENT row
    therefore proves the same unique match as scanning every frozen prompt.
    Non-prefix-free workloads and changed/unbound rows retain the original scan.
    No generated prefix, version, cohort, KV block or proposal is memoized.
    """

    def __init__(self, original):
        super().__init__(original.stable_prompts)
        self._frozen_prompts = MappingProxyType(self.stable_prompts)
        self.stable_prompts = self._frozen_prompts
        ordered = sorted(self.stable_prompts.values())
        # Lexicographic adjacency suffices: descendants of a prefix are contiguous.
        self.prefix_free = all(b[:len(a)] != a for a, b in zip(ordered, ordered[1:]))
        self.internal_to_stable = dict(original.internal_to_stable)
        self.stable_to_internal = dict(original.stable_to_internal)
        self._lock = threading.RLock()
        self._counts = dict.fromkeys(COUNTERS, 0)

    def bind(self, internal_request_id, physical_token_prefix):
        with self._lock:
            start = time.monotonic_ns()
            self._counts["bind_calls"] += 1
            try:
                require(self.stable_prompts is self._frozen_prompts,
                        "fixed identity frozen prompt table was replaced")
                # Keep the original empty-ID, identity-change and alias checks.
                return super().bind(internal_request_id, physical_token_prefix)
            except BaseException:
                self._counts["binding_errors"] += 1
                raise
            finally:
                self._counts["inclusive_host_ns"] += time.monotonic_ns() - start

    def _match_for_binding(self, internal_id, physical_token_prefix):
        c = self._counts
        tokens = tuple(int(item) for item in physical_token_prefix)
        c["linear_candidate_comparisons"] += len(self.stable_prompts)
        previous = self.internal_to_stable.get(internal_id)
        prompt = self.stable_prompts.get(previous)
        if self.prefix_free and prompt is not None:
            c["candidate_comparisons"] += 1
            if tokens[:len(prompt)] == prompt:
                c["validated_binding_reuses"] += 1
                return previous
        c["full_scans"] += 1
        c["candidate_comparisons"] += len(self.stable_prompts)
        return self.match(tokens)

    def report(self):
        with self._lock:
            return {
                "mode": "bound-prefix", "prefix_free": self.prefix_free,
                "frozen_prompt_count": len(self.stable_prompts), **self._counts,
                "scope": "owner lifetime; current full prompt checked on every bind",
                "live_state_cached": False,
            }


def selected_mode():
    mode = os.environ.get("SR_FIXED_IDENTITY_MATCHING", "linear")
    require(mode in MODES, "unknown fixed identity matching mode", actual=mode)
    return mode


def install(owner, attribute):
    """Startup only, before the owner is used. Repeated snapshot RPCs do not rebuild."""
    if selected_mode() == "linear":
        return
    require(bool(os.environ.get("SR_FIXED_POINT")), "identity reuse requires a fixed point")
    original = getattr(owner, attribute)
    if not isinstance(original, BoundPromptIdentityMap):
        require(type(original) is FrozenPromptIdentityMap, "unexpected fixed identity owner")
        setattr(owner, attribute, BoundPromptIdentityMap(original))


def report(identity):
    if isinstance(identity, BoundPromptIdentityMap):
        return identity.report()
    return {"mode": "linear"}


def scheduler_report(scheduler):
    identity = getattr(scheduler, "_resident_identity", None)
    if identity is None:
        identity = getattr(scheduler, "_dual_identity", None)
    return report(identity)


def qualify(runtime, expected):
    """Make selection reviewable in compact results, including zero-access probes."""
    rows = {"scheduler": runtime.get("identity_matching", {"mode": "linear"})}
    for device in runtime.get("target_devices", []):
        rank = device["device"]["identity"]["global_rank"]
        rows["target-rank-" + str(rank)] = device.get("identity_matching", {"mode": "linear"})
    for owner, row in rows.items():
        require(row.get("mode") == expected, "fixed identity matching selection differs",
                field=owner, expected=expected, actual=row.get("mode"))
        if expected == "bound-prefix":
            require(all(type(row.get(k)) is int and row[k] >= 0 for k in COUNTERS),
                    "fixed identity matching counters missing", field=owner)
            require(row["validated_binding_reuses"] + row["full_scans"] <= row["bind_calls"],
                    "fixed identity matching counters inconsistent", field=owner)
    steps = [s for s in runtime.get("target_steps", []) if s.get("window") and s.get("B")]
    if expected == "bound-prefix":
        require(all(s.get("identity_matching", {}).get("mode") == expected for s in steps),
                "fixed measured scheduler matching evidence missing")
    measured = {k: sum(s.get("identity_matching", {}).get(k, 0) for s in steps) for k in COUNTERS}
    return {"mode": expected, "by_owner": rows,
            "measured_scheduler": measured if expected == "bound-prefix" else None,
            "measured_scheduler_steps": len(steps),
            "counter_scope": "startup through final snapshot; step deltas identify measurement"}
