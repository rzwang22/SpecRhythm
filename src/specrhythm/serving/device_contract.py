"""Offline device/verification contracts; snapshots reuse existing worker reads only."""

from specrhythm.serving.common import DataError

PREPOST_MODES = (
    "serial-prepost3",
    "serial-eager-prepost3",
    "pingpong-prepost3",
    "pingpong-eager-prepost3",
    "serial-k3", "pingpong-k3", "pingpong-eager-k3",
)
SCHEMA = "specrhythm.prepost-device-evidence.v1"
SERIAL_PATH = "specrhythm.serving.prepost_proposer.PrePostProposer"
PING_PATH = "specrhythm.serving.ping_prepost_proposer.PingPrePostProposer"


def snapshot_contract(proposer, mode):
    """Record actual loaded class ancestry and existing counters, never query CUDA."""
    mro = [c.__module__ + "." + c.__qualname__ for c in type(proposer).__mro__]
    path = next((p for p in (PING_PATH, SERIAL_PATH) if p in mro), None)
    return dict(
        schema_version=SCHEMA,
        mode=mode,
        proposer_class=mro[0],
        proposer_path=path,
        proposer_tp_rank=proposer.tp_rank,
        verification_hooks=dict(proposer.hooks_seen),
        hook_scope="rank0 per-request start/end; rank1 uses TP/native forward evidence",
        dual_uuid_query=dict(
            status="NOT_APPLICABLE",
            reason="Serial-derived proposer; no DualVerificationUuidQuery consumer",
        ),
        source="s2_runtime._target_capacity_snapshot; existing worker identity snapshot",
    )


class Check:
    def __init__(self, mode, stage):
        self.mode, self.stage = mode, stage

    def fail(self, artifact, rank, field, expected, actual=None):
        raise DataError(
            f"device evidence: mode={self.mode} stage={self.stage} file={artifact} "
            f"rank={rank} field={field}; expected {expected}",
            mode=self.mode,
            stage=self.stage,
            artifact=artifact,
            rank=rank,
            field=field,
            expected=expected,
            actual=actual,
            failure_layer="report_qualification",
        )

    def need(self, row, field, artifact="runtime.json", rank="TP0/1"):
        if not isinstance(row, dict) or field not in row:
            self.fail(artifact, rank, field, "present raw producer evidence")
        return row[field]

    def equal(self, actual, expected, field, artifact="runtime.json", rank="TP0/1"):
        if actual != expected:
            self.fail(artifact, rank, field, expected, actual)

    def ranks(self, rows, artifact, field):
        if not isinstance(rows, list) or len(rows) != 2:
            self.fail(artifact, "TP0/1", field, "exactly two Target rank rows", rows)
        values = [self.need(r, "global_rank", artifact) for r in rows]
        if any(type(v) is not int for v in values):
            self.fail(
                artifact, "TP0/1", field + ".global_rank", "integer TP ranks 0 and 1", values
            )
        self.equal(sorted(values), [0, 1], field + ".global_rank", artifact)
        return {r["global_rank"]: r for r in rows}


def legacy_dual(runtime, mode, stage):
    """Original scan live-query equality, including legitimate zero verification probes."""
    c = Check(mode, stage)
    steps = c.need(runtime, "target_steps")
    verifications = sum(any(row["candidate_positions"] for row in s["rows"]) for s in steps)
    final = c.need(runtime, "target_final_memory")
    c.equal(len(final), 2, "target_final_memory.length")
    ranks = []
    for rank, row in enumerate(final):
        query = c.need(row, "dual_uuid_query", rank=rank)
        expected = dict(
            uuid_query_mode="live",
            uuid_initial_validation_count=1,
            uuid_cache_hit_count=0,
            uuid_verification_access_count=verifications,
            uuid_verification_subprocess_query_count=verifications,
        )
        for key, value in expected.items():
            c.equal(c.need(query, key, rank=rank), value, "dual_uuid_query." + key, rank=rank)
        ranks.append(query)
    return ranks


def qualify_prepost(runtime, backend, actual, mode, *, probe=False, stage=None):
    """Reject incomplete raw evidence with context, including malformed nested data."""
    stage = stage or ("capacity_probe" if probe else "performance")
    try:
        return _qualify_prepost(runtime, backend, actual, mode, probe=probe, stage=stage)
    except DataError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        Check(mode, stage).fail(
            "runtime.json / actual-capacity.json / draft-backend-report.json",
            "Draft+TP0/1",
            str(error),
            "complete typed raw device evidence",
        )


def qualify_run_kind(runtime, mode, *, probe, stage):
    """Strict invocation/report agreement, also for the joint Target-only reference."""
    c = Check(mode, stage)
    selected = c.need(runtime, "point")
    c.equal(c.need(selected, "mode"), mode, "point.mode")
    c.equal(c.need(selected, "runtime_mode"), mode, "point.runtime_mode")
    for field, value in (("execution probe parameter", probe),
                         ("probe", c.need(runtime, "probe")),
                         ("point.probe", c.need(selected, "probe"))):
        if type(value) is not bool:
            c.fail("runtime.json", "TP0/1", field, "actual boolean", value)
        c.equal(value, probe, field)
    scan, correctness = selected.get("scan", False), selected.get("prepost_correctness", False)
    for field, value in (("point.scan", scan), ("point.prepost_correctness", correctness)):
        if type(value) is not bool:
            c.fail("runtime.json", "TP0/1", field, "actual boolean when present", value)
    c.equal(stage, "capacity_probe" if probe else "correctness" if correctness else "performance",
            "stage/point/probe")
    if correctness:
        c.equal((probe, scan), (False, False), "correctness invocation")
        c.equal("decode_scan" in runtime, False, "decode_scan not applicable to correctness")
    c.equal("decode_scan" in runtime, scan, "point.scan/decode_scan")


def _qualify_prepost(runtime, backend, actual, mode, *, probe, stage):
    """Same offline contract for probe, full-output correctness and performance.

    Existing lifecycle/preparation/capacity qualifiers remain responsible for physical
    preparation and release. Here their identity sources are bound to native records.
    """
    from specrhythm.serving.fixed_results import device_batches, rounds_by_prefix

    c = Check(mode, stage)
    c.equal(mode in PREPOST_MODES, True, "supported_mode")
    qualify_run_kind(runtime, mode, probe=probe, stage=stage)
    initial = c.ranks(
        c.need(actual, "target_worker_ranks", "actual-capacity.json"),
        "actual-capacity.json",
        "target_worker_ranks",
    )
    final = c.ranks(c.need(runtime, "target_final_memory"), "runtime.json", "target_final_memory")
    devices = c.need(runtime, "target_devices")
    identities = [c.need(c.need(d, "device"), "identity") for d in devices]
    native = c.ranks(identities, "runtime.json", "target_devices.device.identity")
    steps = c.need(runtime, "target_steps")
    verification_rows = sum(bool(r["candidate_positions"]) for s in steps for r in s["rows"])
    if probe:
        c.equal(steps, [], "target_steps")
        c.equal(c.need(runtime, "stop_reason"), "capacity_probe", "stop_reason")
    elif not verification_rows:
        c.fail("runtime.json", "TP0/1", "target_steps", "nonzero actual decode verification")
    path = PING_PATH if mode.startswith("pingpong-") or mode == "serial-k3" else SERIAL_PATH
    uuids = []
    for rank in (0, 1):
        start = initial[rank]
        for row, artifact, at_final in (
            (start, "actual-capacity.json", False),
            (final[rank], "runtime.json", True),
        ):
            c.equal(c.need(row, "world_size", artifact, rank), 2, "world_size", artifact, rank)
            c.equal(
                c.need(row, "physical_gpu_id", artifact, rank),
                rank + 1,
                "physical_gpu_id",
                artifact,
                rank,
            )
            uuid = c.need(row, "gpu_uuid", artifact, rank)
            if not isinstance(uuid, str) or not uuid:
                c.fail(artifact, rank, "gpu_uuid", "nonempty actual UUID", uuid)
            c.equal(
                uuid,
                c.need(start, "gpu_uuid", "actual-capacity.json", rank),
                "startup/final UUID binding",
                artifact,
                rank,
            )
            c.equal(
                c.need(row, "all_parameters_on_expected_device", artifact, rank),
                True,
                "all_parameters_on_expected_device",
                artifact,
                rank,
            )
            for key in ("parameter_count", "parameter_bytes", "allocated_memory_bytes"):
                value = c.need(row, key, artifact, rank)
                if type(value) is not int or value <= 0:
                    c.fail(artifact, rank, key, "positive actual loaded model allocation", value)
            contract = c.need(row, "device_evidence_contract", artifact, rank)
            for key, value in (
                ("schema_version", SCHEMA),
                ("mode", mode),
                ("proposer_path", path),
                ("proposer_tp_rank", rank),
            ):
                c.equal(c.need(contract, key, artifact, rank), value, key, artifact, rank)
            hooks = c.need(contract, "verification_hooks", artifact, rank)
            count = verification_rows if at_final and rank == 0 else 0
            for key in ("verify_start", "verify_end"):
                c.equal(
                    c.need(hooks, key, artifact, rank),
                    count,
                    "verification_hooks." + key,
                    artifact,
                    rank,
                )
            na = c.need(contract, "dual_uuid_query", artifact, rank)
            c.equal(
                c.need(na, "status", artifact, rank),
                "NOT_APPLICABLE",
                "dual_uuid_query.status",
                artifact,
                rank,
            )
            c.equal(
                c.need(na, "reason", artifact, rank),
                "Serial-derived proposer; no DualVerificationUuidQuery consumer",
                "dual_uuid_query.reason",
                artifact,
                rank,
            )
            cap = c.need(row, "s2_capacity", artifact, rank)
            for key, value in (
                ("mode", mode),
                ("role", "target"),
                ("physical_gpu_id", rank + 1),
                ("gpu_uuid", uuid),
            ):
                c.equal(
                    c.need(cap, key, artifact, rank), value, "s2_capacity." + key, artifact, rank
                )
        for key in ("global_rank", "local_rank", "physical_gpu_id", "gpu_uuid"):
            c.equal(
                c.need(native[rank], key, rank=rank),
                c.need(start, key, rank=rank),
                "native/startup." + key,
                rank=rank,
            )
        c.equal(c.need(native[rank], "role", rank=rank), "target", "native.role", rank=rank)
        uuids.append(start["gpu_uuid"])
    draft = c.need(
        c.need(backend, "fixed_device", "draft-backend-report.json"),
        "identity",
        "draft-backend-report.json",
        "Draft",
    )
    caps = c.need(actual, "ranks", "actual-capacity.json")
    c.equal(len(caps), 3, "ranks.length", "actual-capacity.json")
    c.equal(
        sorted((c.need(r, "role"), c.need(r, "physical_gpu_id")) for r in caps),
        [("draft", 0), ("target", 1), ("target", 2)],
        "ranks.role/device",
        "actual-capacity.json",
    )
    for cap in caps:
        role, gpu = cap["role"], cap["physical_gpu_id"]
        ident = draft if role == "draft" else native[gpu - 1]
        for key, value in (
            ("role", role),
            ("physical_gpu_id", gpu),
            ("gpu_uuid", c.need(cap, "gpu_uuid", "actual-capacity.json", role)),
        ):
            c.equal(c.need(ident, key, rank=role), value, "capacity/native." + key, rank=role)
        c.equal(c.need(cap, "mode"), mode, "capacity.mode", "actual-capacity.json", role)
        for key in ("block_size", "num_gpu_blocks"):
            value = c.need(cap, key, "actual-capacity.json", role)
            if type(value) is not int or value <= 0:
                c.fail("actual-capacity.json", role, key, "positive physical capacity", value)
    uuids.append(c.need(draft, "gpu_uuid", "draft-backend-report.json", "Draft"))
    if any(not isinstance(u, str) or not u for u in uuids) or len(set(uuids)) != 3:
        c.fail(
            "runtime.json / actual-capacity.json",
            "Draft+TP0/1",
            "gpu_uuid",
            "three distinct within-run actual device bindings",
            uuids,
        )
    try:
        joined = device_batches(devices, steps)
        rounds = rounds_by_prefix([r for d in devices for r in d["rounds"]])
        diagnostics = {
            (r["request_id"], r["context_length"]): r for d in devices for r in d["target_rows"]
        }
        for step in steps:
            for row in step["rows"]:
                if not row["candidate_positions"]:
                    continue
                key = row["request_id"], row["context_length"]
                d, r = diagnostics.get(key), rounds.get(key)
                if d is None or r is None:
                    c.fail(
                        "runtime.json",
                        "TP0",
                        "target_rows/rounds",
                        "actual request/prefix proposal association",
                        key,
                    )
                c.equal(
                    len(d["proposal_token_ids"]),
                    row["candidate_positions"],
                    "actual candidate length",
                )
                c.equal(
                    d["proposal_token_ids"],
                    r["proposal_token_ids"],
                    "diagnostic/round proposal tokens",
                )
                c.equal(d["structural_errors"], [], "target_rows.structural_errors")
    except (KeyError, DataError) as error:
        if isinstance(error, DataError) and error.details.get("failure_layer"):
            raise
        c.fail(
            "runtime.json",
            "TP0/1",
            str(error),
            "complete native request/proposal association",
            getattr(error, "details", None),
        )
    return dict(
        schema_version=SCHEMA,
        mode=mode,
        stage=stage,
        status="PASS",
        target_ranks=[0, 1],
        target_physical_gpus=[1, 2],
        draft_physical_gpu=0,
        startup_final_native_binding="MATCHED_WITHIN_RUN",
        verification_requests=verification_rows,
        verification_steps=len(joined),
        dual_uuid_query=dict(
            status="NOT_APPLICABLE",
            reason="Serial-derived proposer; checked snapshots, hooks and native forwards instead",
        ),
        source_files=["actual-capacity.json", "runtime.json", "draft-backend-report.json"],
    )
