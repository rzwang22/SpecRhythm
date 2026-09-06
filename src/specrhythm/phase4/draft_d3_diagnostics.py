"""Explicit D3-only observation. Never imported or installed by a serving backend.

The observer reads CPU/GPU metadata and logits, but never edits request tensors,
page allocation, scheduling, sampling, or returned logits. All records stay in
memory until draft_gate writes one exclusive final artifact. GPU reads perturb
timing, so this is ineligible for performance interpretation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from specrhythm.phase4.draft_batch import mapped_rows, unique_ids
from specrhythm.phase4.serial import token_prefix_hash


def assert_private_blocks(owners: Mapping[str, Sequence[Sequence[int]]]) -> None:
    """Private full-attention pages cannot be shared across or within requests."""
    seen = {}
    for rid, groups in owners.items():
        for group, blocks in enumerate(groups):
            for block in blocks:
                if type(block) is not int or block < 0:
                    raise RuntimeError("D3 invalid external KV block ID")
                key = (group, block)
                if key in seen:
                    raise RuntimeError(f"D3 private KV alias: {seen[key]} and {rid}: {key}")
                seen[key] = rid


def authoritative_logit_ids(ids, query_start, logits_indices) -> tuple[str, ...]:
    """Non-spec MRV1: each logit is hidden_states[query_start_loc[i+1]-1]."""
    unique_ids(ids)
    if (
        len(query_start) != len(ids) + 1
        or query_start[0] != 0
        or any(b <= a for a, b in zip(query_start, query_start[1:]))
        or list(logits_indices) != [end - 1 for end in query_start[1:]]
    ):
        raise RuntimeError("D3 non-speculative logits row domain is not one final row per request")
    return tuple(ids)


def assert_runner_rows(planned, observed, external) -> None:
    by_id = mapped_rows(list(planned), [r["internal_id"] for r in observed], observed)
    assert_private_blocks(external)
    for rid, row in by_id.items():
        plan = planned[rid]
        expected_context = token_prefix_hash(plan.context)
        if (
            row["context_sha256"] != expected_context
            or row["runner_context_sha256"] != expected_context
            or row["computed_tokens"] != plan.valid_length
            or row["runner_computed_tokens"] != plan.valid_length
            or row["prompt_tokens"] != len(plan.context)
            or row["runner_prompt_tokens"] != len(plan.context)
            or row["active_token_count"] != len(plan.context)
            or row["scheduled_tokens"] != len(plan.suffix)
            or row["scheduled_input_token_ids"] != list(plan.suffix)
            or row["block_ids"] != external[rid]
        ):
            raise RuntimeError(f"D3 runner row/context/frontier/page-table mismatch: {rid}")


class D3MaterializationDiagnostic:
    def __init__(self, backend: Any) -> None:
        self.backend, self.worker = backend, backend.worker
        self.runner = self.worker.runner
        self.frames: list[dict] = []
        self.frame = None
        self.planned = {}
        self.patches = []
        self.hook = None
        self.round = None
        self.gpu_read_count = 0
        self.attached = False

    def _read(self, tensor):
        self.gpu_read_count += 1
        return tensor.detach().cpu().tolist()

    def _owners(self):
        owners = {
            rid: [list(g) for g in self.worker.kv.get_block_ids(rid)] for rid in self.worker.views
        }
        if self.frame is not None:
            self.frame["last_external_owners"] = owners
        assert_private_blocks(owners)
        return owners

    def _stable(self, internal_id):
        identities = {state.internal_id: rid for rid, state in self.backend.states.items()}
        return identities[internal_id]

    def _patch(self, obj, name, wrapper):
        previous = obj.__dict__.get(name)
        existed = name in obj.__dict__
        setattr(obj, name, wrapper(getattr(obj, name)))
        self.patches.append((obj, name, existed, previous))

    def attach(self) -> None:
        if self.attached:
            raise RuntimeError("D3 observer already installed")
        # Full pinned runner/InputBatch/block-table files are already guarded at
        # construction. These seams must also exist on this actual worker.
        for obj, name in (
            (self.worker, "materialize"),
            (self.runner, "_update_states"),
            (self.runner, "_prepare_inputs"),
            (self.worker.model, "compute_logits"),
        ):
            if not callable(getattr(obj, name, None)):
                raise RuntimeError(f"D3 observer requires pinned method: {name}")
        self._patch(self.worker, "materialize", self._wrap_materialize)
        self._patch(self.runner, "_update_states", self._wrap_update)
        self._patch(self.runner, "_prepare_inputs", self._wrap_prepare)
        self._patch(self.worker.model, "compute_logits", self._wrap_logits)
        self.hook = self.worker.model.register_forward_pre_hook(self._before_model)
        self.attached = True

    def close(self) -> None:
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        for obj, name, existed, previous in reversed(self.patches):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.patches.clear()
        self.attached = False

    def _wrap_materialize(self, original):
        def observed(rows, purpose):
            if not rows:
                return original(rows, purpose)
            frame = {
                "index": len(self.frames),
                "round": self.round,
                "purpose": purpose,
                "checks_passed": False,
                "errors": [],
            }
            self.frames.append(frame)
            self.frame = frame
            self.planned = {r.request_id: r for r in rows}
            try:
                owners = self._owners()
                frame["before_external_owners"] = owners
                frame["before_materialize"] = [
                    {
                        "request_id": self._stable(r.request_id),
                        "internal_id": r.request_id,
                        "context_length": len(r.context),
                        "valid_length": r.valid_length,
                        "context_sha256": token_prefix_hash(r.context),
                        "suffix": list(r.suffix),
                        "external_block_ids": owners[r.request_id],
                        "external_block_count": sum(map(len, owners[r.request_id])),
                    }
                    for r in rows
                ]
                # The diagnostic cohort's contexts are distinct. Shared initial
                # substrings are legal; whole-context identity here is not.
                hashes = [r["context_sha256"] for r in frame["before_materialize"]]
                if len(set(hashes)) != len(hashes):
                    raise RuntimeError("D3 distinct fixture requests acquired identical contexts")
                result = original(rows, purpose)
                ids = tuple(self.runner.input_batch.req_ids)
                source_ids = frame["after_forward"]["authoritative_logits_request_ids"]
                if ids != tuple(source_ids):
                    raise RuntimeError(
                        "D3 completion InputBatch changed after logits construction"
                    )
                keys = tuple(result)
                returned = self._read(
                    self.worker.torch.stack([result[r] for r in keys]).argmax(-1)
                )
                source_top1 = dict(zip(source_ids, frame["after_forward"]["top1_token_ids"]))
                expected = mapped_rows(list(self.planned), list(keys), returned)
                frame["completion_mapping"] = {
                    "actual_input_batch_ids": list(ids),
                    "returned_ids": list(keys),
                    "top1_by_id": expected,
                }
                if expected != source_top1:
                    raise RuntimeError(
                        "D3 complete_draft_forward associated logits with wrong IDs"
                    )
                frame["after_external_owners"] = self._owners()
                frame["checks_passed"] = True
                return result
            except Exception as error:
                frame["errors"].append(f"{type(error).__name__}: {error}")
                raise
            finally:
                self.frame, self.planned = None, {}

        return observed

    def _snapshot_rows(self, scheduler_output):
        batch = self.runner.input_batch
        ids = tuple(batch.req_ids)
        actual = []
        for i, rid in enumerate(ids):
            context_len = int(batch.num_prompt_tokens[i])
            computed = int(batch.num_computed_tokens_cpu[i])
            scheduled = int(scheduler_output.num_scheduled_tokens[rid])
            blocks = []
            for table in batch.block_table.block_tables:
                # Current Plan A uses the identical allocation/kernel block size.
                # A split layout must be explicitly audited before this observer
                # may compare native page IDs directly with kernel page IDs.
                if table.blocks_per_kv_block != 1:
                    raise RuntimeError("D3 observer requires unsplit kernel block IDs")
                n = int(table.num_blocks_per_row[i])
                blocks.append(table.get_numpy_array()[i, :n].tolist())
            state = self.runner.requests[rid]
            actual.append(
                {
                    "request_id": self._stable(rid),
                    "internal_id": rid,
                    "row_index": i,
                    "computed_tokens": computed,
                    "runner_computed_tokens": state.num_computed_tokens,
                    "prompt_tokens": context_len,
                    "runner_prompt_tokens": state.num_prompt_tokens,
                    "active_token_count": batch._get_active_token_count(i),
                    "context_sha256": token_prefix_hash(
                        batch.token_ids_cpu[i, :context_len].tolist()
                    ),
                    "runner_context_sha256": token_prefix_hash(state.prompt_token_ids),
                    "block_ids": blocks,
                    "scheduled_tokens": scheduled,
                    "scheduled_input_token_ids": batch.token_ids_cpu[
                        i, computed : computed + scheduled
                    ].tolist(),
                }
            )
        row_map = dict(batch.req_id_to_index)
        self.frame["after_update_states"] = actual
        self.frame["req_id_to_index"] = row_map
        if row_map != {rid: i for i, rid in enumerate(ids)}:
            raise RuntimeError("D3 InputBatch ID-to-index map differs from row IDs")
        owners = self._owners()
        self.frame["allocated_external_owners"] = owners
        assert_runner_rows(self.planned, actual, owners)
        return actual

    def _wrap_update(self, original):
        def observed(scheduler_output):
            result = original(scheduler_output)
            if self.frame is not None:
                self.frame["after_update_states"] = self._snapshot_rows(scheduler_output)
            return result

        return observed

    def _wrap_prepare(self, original):
        def observed(scheduler_output, num_scheduled_tokens):
            result = original(scheduler_output, num_scheduled_tokens)
            if self.frame is not None:
                n = self.runner.input_batch.num_reqs
                starts = self._read(self.runner.query_start_loc.gpu[: n + 1])
                indices = self._read(result[0])
                ids = tuple(self.runner.input_batch.req_ids)
                self.frame["prepared_logits_domain"] = {
                    "query_start_loc": starts,
                    "logits_indices": indices,
                    "request_ids": list(ids),
                    "spec_decode_metadata_present": result[1] is not None,
                }
                authoritative_logit_ids(ids, starts, indices)
                if result[1] is not None:
                    raise RuntimeError("D3 observer received speculative logits metadata")
            return result

        return observed

    def _before_model(self, _model, _args):
        if self.frame is None:
            return
        runner, batch = self.runner, self.runner.input_batch
        domain = self.frame["prepared_logits_domain"]
        ids, starts = domain["request_ids"], domain["query_start_loc"]
        self.frame["immediately_before_model_input_batch_ids"] = list(batch.req_ids)
        if tuple(batch.req_ids) != tuple(ids):
            raise RuntimeError("D3 row domain changed between input preparation and model forward")
        count = starts[-1]
        inputs = self._read(runner.input_ids.gpu[:count])
        positions = self._read(runner.positions[:count])
        computed = self._read(runner.num_computed_tokens[: len(ids)])
        lengths = self._read(runner.seq_lens[: len(ids)])
        row_indices = self._read(runner.req_indices.gpu[:count])
        slots = [self._read(t.slot_mapping.gpu[:count]) for t in batch.block_table.block_tables]
        gpu_blocks = [
            self._read(t.get_device_tensor(len(ids))) for t in batch.block_table.block_tables
        ]
        rows = []
        failures = []
        for i, rid in enumerate(ids):
            plan = self.planned[rid]
            lo, hi = starts[i : i + 2]
            expected_positions = list(range(plan.valid_length, len(plan.context)))
            page_ids = self.frame["allocated_external_owners"][rid]
            expected_slots = [
                [
                    group[pos // t.block_size] * t.block_size + pos % t.block_size
                    for pos in expected_positions
                ]
                for group, t in zip(page_ids, batch.block_table.block_tables)
            ]
            row = {
                "internal_id": rid,
                "request_id": self._stable(rid),
                "row_index": i,
                "input_ids": inputs[lo:hi],
                "positions": positions[lo:hi],
                "computed_tokens": computed[i],
                "seq_len": lengths[i],
                "req_indices": row_indices[lo:hi],
                "block_ids": [
                    table[i][: len(group)] for table, group in zip(gpu_blocks, page_ids)
                ],
                "slot_mapping": [group[lo:hi] for group in slots],
            }
            rows.append(row)
            self.frame["immediately_before_model"] = rows
            if (
                row["input_ids"] != list(plan.suffix)
                or row["positions"] != expected_positions
                or computed[i] != plan.valid_length
                or lengths[i] != len(plan.context)
                or row["req_indices"] != [i] * len(plan.suffix)
                or row["block_ids"] != page_ids
                or row["slot_mapping"] != expected_slots
            ):
                failures.append(rid)
        if failures:
            raise RuntimeError(f"D3 prepared GPU input/frontier/page/slot mismatch: {failures}")

    def _wrap_logits(self, original):
        def observed(hidden_states, *args, **kwargs):
            ids = tuple(self.runner.input_batch.req_ids)
            logits = original(hidden_states, *args, **kwargs)
            if self.frame is not None:
                domain = self.frame["prepared_logits_domain"]["request_ids"]
                self.frame["logits_construction"] = {
                    "input_batch_ids": list(ids),
                    "logits_shape": list(logits.shape),
                }
                if ids != tuple(domain) or logits.shape[0] != len(ids):
                    raise RuntimeError("D3 logits construction changed authoritative row domain")
                values, tokens = logits.topk(2, dim=-1)
                token_rows, value_rows = self._read(tokens), self._read(values.float())
                self.frame["after_forward"] = {
                    "authoritative_logits_request_ids": list(ids),
                    "authority": "post-update IDs + GPU query_start_loc/logits_indices",
                    # topk tie order need not match argmax; use the same greedy
                    # operation as the real Draft selector for authoritative top1.
                    "top1_token_ids": self._read(logits.argmax(dim=-1)),
                    "top2_token_ids": token_rows,
                    "top2_logits": value_rows,
                    "top1_minus_top2": [a - b for a, b in value_rows],
                }
            return logits

        return observed

    def report(self) -> dict:
        return {
            "schema_version": "specrhythm.phase4b3-d3-materialization-diagnostic.v1",
            "diagnostic_only": True,
            "performance_result": False,
            "serving_path_instrumented": False,
            "gpu_read_count": self.gpu_read_count,
            "root_cause_proven": False,
            "frames": self.frames,
            "observer_removed": not self.attached,
            "all_structural_checks_passed": bool(self.frames)
            and all(frame["checks_passed"] for frame in self.frames),
        }
