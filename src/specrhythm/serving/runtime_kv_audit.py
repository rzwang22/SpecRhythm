"""Incremental private-KV evidence at the sole Draft allocator mutation boundary.

No cached global PASS: touched handles and current allocator tables are compared
on every check. Allocation/free calls are scoped to the current worker operation;
reverse ownership is updated before a forward can use new blocks. Full-attention,
no-prefix-cache Draft is required by the existing real worker contract.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from specrhythm.continuation.trace import TRACE
from specrhythm.serving.common import require
from specrhythm.serving.s2_pool import prefix_record


class AllocatorBoundary:
    def __init__(self, allocator, guard):
        self.original, self.guard = allocator, guard

    def __getattr__(self, name):
        # The pinned worker uses only these non-ownership-mutating APIs. A new
        # allocator API must be reviewed here instead of bypassing the ledger.
        if name not in ("get_block_ids", "new_step_starts", "take_new_block_ids"):
            raise AttributeError("runtime audit does not cover allocator API: " + name)
        return getattr(self.original, name)

    def allocate_slots(self, request, *args, **kwargs):
        g, rid = self.guard, request.request_id
        g.check_owner()
        require(
            g.operation == "write" and rid in g.scope,
            "allocator allocation outside declared Draft write",
        )
        result = self.original.allocate_slots(request, *args, **kwargs)
        if result is not None:
            g.observe_blocks(rid, growth=True)
        return result

    def free(self, request, *args, **kwargs):
        g, rid = self.guard, request.request_id
        g.check_owner()
        require(
            g.operation == "release" and rid in g.scope and not g.pending,
            "allocator free outside fenced Draft release",
        )
        g.observe_blocks(rid)
        result = self.original.free(request, *args, **kwargs)
        g.retire(rid)
        return result


class RuntimeKVGuard:
    def __init__(self, backend):
        self.backend, self.worker = backend, backend.worker
        self.blocks, self.owners, self.written, self.handles = {}, {}, {}, {}
        self.released = set()
        self.operation, self.scope, self.pending = None, set(), False
        self.checks = self.visits = self.prefix_visits = self.mutations = self.peak_blocks = 0
        self.allocations = self.releases = 0
        self.block_check_ns = 0
        kv = self.worker.kv
        require(
            callable(getattr(kv, "allocate_slots", None)) and callable(getattr(kv, "free", None)),
            "runtime audit requires observable allocator allocate_slots/free boundaries",
        )
        self.worker.kv = AllocatorBoundary(kv, self)
        materialize, fence, release = (
            self.worker.materialize,
            self.worker.fence,
            self.worker.release,
        )

        def guarded_materialize(rows, purpose):
            self.check_owner()
            if not rows:
                return materialize(rows, purpose)
            with TRACE.span("runtime_write_check", affected_requests=len(rows), purpose=purpose):
                packet = backend.audit_control()
                if purpose != "setup":
                    # One owner-local snapshot/check event per compatible write batch.
                    self.check(tuple(r.request_id.removeprefix("sr-draft:") for r in rows), packet)
                for row in rows:
                    rid = row.request_id.removeprefix("sr-draft:")
                    if purpose == "setup":
                        require(
                            backend.audit_setup_allowed(rid, packet),
                            "runtime audit forbids timed re-prefill",
                        )
                        require(
                            row.request_id not in self.blocks
                            and row.request_id not in self.released
                            and row.valid_length == 0,
                            "runtime audit forbids repeated private setup",
                        )
                    else:
                        require(
                            packet["requests"][rid]["state"]
                            in ("ACTIVE", "FINISHED"),
                            "runtime write requires legal admitted state",
                        )
                        # Initial proposal/enrollment still requires ACTIVE in
                        # S2/GPU admission. FINISHED may arrive during an already
                        # admitted batch; only authoritative owner feedback can
                        # settle/cancel it. Do not turn that legal race into failure.
                        require(
                            0 <= row.valid_length <= self.written[row.request_id],
                            "runtime KV write frontier exceeds materialized evidence",
                        )
                    require(
                        len(row.context) <= backend.max_model_len,
                        "runtime KV write exceeds model capacity",
                    )
            with self.transaction("write", [r.request_id for r in rows]):
                self.pending = True
                result = materialize(rows, purpose)
                for row in rows:
                    self.observe_blocks(row.request_id)
                    self.written[row.request_id] = len(row.context)
                    self._capacity(row.request_id, len(row.context))
            return result

        def guarded_fence(reason):
            result = fence(reason)
            self.pending = False  # Only a successful original fence clears writes.
            return result

        def guarded_release(ids):
            require(not self.pending, "runtime release before completed write fence")
            with self.transaction("release", ids):
                result = release(ids)
                require(
                    all(rid in self.released for rid in ids),
                    "runtime release bypassed allocator ownership boundary",
                )
                return result

        self.worker.materialize = guarded_materialize
        self.worker.fence = guarded_fence
        self.worker.release = guarded_release

    def check_owner(self):
        require(
            threading.get_ident() == self.backend.owner_thread and not self.backend.closed,
            "runtime allocator used outside live Draft owner thread",
        )

    @contextmanager
    def transaction(self, operation, ids):
        self.check_owner()
        require(self.operation is None, "nested allocator mutation")
        self.operation, self.scope = operation, set(ids)
        started, before = time.monotonic_ns(), self.block_check_ns
        try:
            yield
        finally:
            self.operation, self.scope = None, set()
            TRACE.event(
                "runtime_allocator_operation",
                start_ns=started,
                end_ns=time.monotonic_ns(),
                operation=operation,
                affected_requests=len(ids),
                allocator_check_ns=self.block_check_ns - before,
            )

    def _capacity(self, internal, frontier):
        b = self.backend
        require(
            type(frontier) is int and 0 < frontier <= b.max_model_len,
            "runtime invalid materialized frontier",
        )
        require(
            all(len(g) * b.provenance["block_size"] >= frontier for g in self.blocks[internal]),
            "runtime frontier exceeds allocated KV capacity",
        )

    def observe_blocks(self, internal, *, growth=False):
        started = time.monotonic_ns()
        try:
            return self._observe_blocks(internal, growth=growth)
        finally:
            self.block_check_ns += time.monotonic_ns() - started

    def _observe_blocks(self, internal, *, growth=False):
        groups = tuple(tuple(g) for g in self.worker.kv.get_block_ids(internal))
        require(groups and all(groups), "runtime missing private block table")
        old = self.blocks.get(internal)
        if old is not None:
            require(
                len(old) == len(groups)
                and all(
                    (new[: len(previous)] == previous if growth else new == previous)
                    for previous, new in zip(old, groups)
                ),
                "runtime allocator ownership changed without growth/release",
            )
        else:
            require(
                growth and internal not in self.released,
                "runtime unknown or released allocator request",
            )
        added = []
        for group, blocks in enumerate(groups):
            require(len(set(blocks)) == len(blocks), "runtime duplicate KV block")
            for block in blocks:
                require(
                    type(block) is int
                    and 0 <= block < self.backend.provenance["kv_cache_num_blocks"],
                    "runtime KV block outside allocator capacity",
                )
                key = group, block
                require(
                    key not in self.owners or self.owners[key] == internal,
                    "runtime KV block ownership conflict",
                )
                if key not in self.owners:
                    added.append(key)
        for key in added:
            self.owners[key] = internal
        self.blocks[internal] = groups
        self.allocations += len(added)
        self.peak_blocks = max(self.peak_blocks, len(self.owners))
        self.mutations += int(bool(added))

    def retire(self, internal):
        for group, blocks in enumerate(self.blocks.pop(internal)):
            for block in blocks:
                require(
                    self.owners.pop((group, block)) == internal,
                    "runtime free has wrong block owner",
                )
                self.releases += 1
        self.written.pop(internal, None)
        self.handles.pop(internal.removeprefix("sr-draft:"), None)
        self.released.add(internal)
        self.mutations += 1
        TRACE.event("runtime_allocator_release", request_id=internal, mutation=self.mutations)

    def row(self, rid):
        state = self.backend.states[rid]
        old = self.handles.get(rid)
        if old is not None:
            prefix, version, hashed = old
            require(state.next_round >= version, "runtime stale prefix version")
            require(
                (state.next_round == version and state.prefix == prefix)
                or (state.next_round == version + 1 and state.prefix[: len(prefix)] == prefix),
                "runtime unversioned or conflicting committed prefix",
            )
        if old is None or state.prefix is not old[0]:
            result = prefix_record(
                state.prefix, state.materialized, self.blocks[state.internal_id]
            )
            self.prefix_visits += 1
        else:
            result = dict(
                prefix_sha256=hashed,
                materialized_tokens=state.materialized,
                block_ids=[list(g) for g in self.blocks[state.internal_id]],
            )
        self.handles[rid] = state.prefix, state.next_round, result["prefix_sha256"]
        return result

    @TRACE.observe("runtime_resident_check")
    def check(self, ids, packet):
        self.check_owner()
        self.checks += 1
        initial = self.backend.pool.initial or {}
        for rid in ids:
            self.visits += 1
            if rid in self.backend.retired:
                require(
                    "sr-draft:" + rid in self.released and rid not in self.backend.states,
                    "runtime retired KV resurrected",
                )
                continue
            state = self.backend.states[rid]
            require(
                state.internal_id == "sr-draft:" + rid
                and type(state.next_round) is int
                and state.next_round >= 0
                and type(state.prefix) is tuple,
                "runtime invalid request handle identity/version/immutable prefix",
            )
            status = packet["requests"][rid]["state"]
            require(
                status in ("STAGED", "QUEUED", "ACTIVE", "FINISHED"),
                "runtime illegal request state",
            )
            self.observe_blocks(state.internal_id)
            self._capacity(state.internal_id, state.materialized)
            require(
                state.materialized >= len(state.prefix),
                "runtime frontier precedes committed prefix",
            )
            old = self.handles.get(rid)
            require(
                state.materialized == self.written[state.internal_id]
                or (
                    old is not None
                    and state.next_round == old[1] + 1
                    and state.materialized <= self.written[state.internal_id]
                ),
                "runtime materialized frontier lacks write/settlement evidence",
            )
            self.written[state.internal_id] = state.materialized
            row = self.row(rid)
            if rid in initial:
                base = initial[rid]
                if status in ("STAGED", "QUEUED"):
                    require(row == base, "runtime staged/queued prefix changed or evicted")
                elif status == "ACTIVE":
                    require(
                        row["materialized_tokens"] >= base["materialized_tokens"]
                        and all(
                            row["block_ids"][g][: len(blocks)] == blocks
                            for g, blocks in enumerate(base["block_ids"])
                        ),
                        "runtime active initial prefix replaced/re-prefilled",
                    )
        self.backend.pool.peak_blocks = max(self.backend.pool.peak_blocks, self.peak_blocks)
        TRACE.event(
            "runtime_checked_requests",
            affected_requests=len(ids),
            allocator_mutation=self.mutations,
        )

    def reconcile(self, physical):
        require(set(physical) == set(self.backend.states), "runtime full scope mismatch")
        require(
            set(self.blocks) == {s.internal_id for s in self.backend.states.values()},
            "runtime lifecycle lost live allocator request",
        )
        require(
            len(self.owners) == sum(len(g) for row in physical.values() for g in row["block_ids"]),
            "runtime block ledger differs from complete audit",
        )
        for rid, row in physical.items():
            require(row == self.row(rid), "runtime incremental row differs from complete audit")

    def report(self):
        return dict(
            checks=self.checks,
            affected_request_checks=self.visits,
            changed_prefix_hash_visits=self.prefix_visits,
            mutations=self.mutations,
            blocks_allocated=self.allocations,
            blocks_freed=self.releases,
            peak_blocks=self.peak_blocks,
            live_blocks=len(self.owners),
            live_requests=len(self.blocks),
            pending_write=self.pending,
            coverage="all worker allocate_slots/free and materialize/fence boundaries",
        )
