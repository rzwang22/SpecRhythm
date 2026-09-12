"""Physical allocator evidence and in-place GPU residency guards (no KV tensor serialization)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from specrhythm.phase4.serial import token_prefix_hash
from specrhythm.serving.common import read_json, require


def control():
    return read_json(Path(os.environ["SR_S2_CONTROL"]))


def publish(path, value):
    """Single-writer control snapshot; atomic visibility, no per-step fsync."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, allow_nan=False)
    temporary.replace(path)


def block_count(rows):
    return sum(len(g) for r in rows.values() for g in r["block_ids"])


class ResidentPoolAudit:
    def __init__(self, role):
        self.role = role
        self.initial = None
        self.peak_blocks = 0
        self.checks = 0

    def check(self, rows, states, *, freeze=False):
        owners = {}
        for rid, row in rows.items():
            require(
                row["materialized_tokens"] > 0 and row["block_ids"],
                "resident prefix has no physical KV",
                role=self.role,
                request_id=rid,
            )
            for group, blocks in enumerate(row["block_ids"]):
                require(
                    blocks and len(set(blocks)) == len(blocks), "invalid private KV block table"
                )
                for block in blocks:
                    require(
                        type(block) is int and block >= 0 and (group, block) not in owners,
                        "resident KV blocks shared across requests",
                        role=self.role,
                        request_id=rid,
                    )
                    owners[group, block] = rid
        if freeze:
            require(self.initial is None, "resident pool cannot be restored over continuation KV")
            self.initial = json.loads(json.dumps(rows))
        elif self.initial is not None:
            for rid, initial in self.initial.items():
                state = states[rid]
                if state["state"] in ("STAGED", "QUEUED"):
                    require(
                        rows.get(rid) == initial,
                        "unarrived/queued resident prefix changed or evicted",
                        role=self.role,
                        request_id=rid,
                    )
                elif state["state"] == "ACTIVE":
                    current = rows.get(rid)
                    require(
                        current is not None
                        and current["materialized_tokens"] >= initial["materialized_tokens"],
                        "active resident KV missing/re-prefilled",
                        role=self.role,
                        request_id=rid,
                    )
                    require(
                        all(
                            current["block_ids"][g][: len(ids)] == ids
                            for g, ids in enumerate(initial["block_ids"])
                        ),
                        "active initial prefix blocks replaced",
                        role=self.role,
                        request_id=rid,
                    )
        self.peak_blocks = max(self.peak_blocks, block_count(rows))
        self.checks += 1

    def report(self):
        return {
            "role": self.role,
            "initial_requests": len(self.initial or {}),
            "initial_blocks": block_count(self.initial or {}),
            "peak_blocks": self.peak_blocks,
            "resident_checks": self.checks,
            "initial": self.initial,
            "restore_method": "fresh engine; real prefill; private GPU KV retained in allocator",
            "snapshot_tensor_copies": 0,
            "timed_reprefill_allowed": False,
        }


def prefix_record(tokens, materialized, blocks):
    return {
        "prefix_sha256": token_prefix_hash(tokens),
        "materialized_tokens": int(materialized),
        "block_ids": [list(g) for g in blocks],
    }
