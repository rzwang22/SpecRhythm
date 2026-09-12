"""Bounded, read-only export of existing Serial/eager evidence; never loads CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_COMMIT = "4a6725b054990e47b7e9c4cf63f38b995d029856"
MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
MAX_ROWS = 100000
MAX_CYCLES = 8
CYCLE_ROWS = 512
FIELDS = frozenset((
    "schema_version", "request_id", "request_ids", "internal_request_ids", "batch_request_ids",
    "target_batch_request_ids", "work_id", "work_ids", "proposal_id", "source_continuation_id",
    "owner_id", "round_id", "round_ids", "step", "B", "Q", "purpose", "phase", "operation",
    "category", "window", "cohort", "terminal", "event", "direction", "request_payload_bytes",
    "response_payload_bytes", "parent_prefix_len", "prefix_token_count", "prefix_version",
    "accepted_draft_tokens", "rejected_draft_tokens", "candidate_positions", "base_root_positions",
    "context_length", "context_lengths", "budgets", "candidate_lengths",
    "first_token_from_cached_logits", "success",
    "worker_fenced", "timeline", "counter_delta", "decision", "rows", "materialized_positions",
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


class Reader:
    def __init__(self, root, file_bytes, total_bytes, rows):
        self.root, self.file_bytes = root, file_bytes
        self.total_bytes, self.rows = total_bytes, rows
        self.used, self.inventory, self.cache = 0, {}, {}

    def read(self, path, *, required=False, jsonl=False):
        path = Path(path)
        require(path.resolve().is_relative_to(self.root), "input escaped source root")
        name = str(path.relative_to(self.root))
        if name in self.cache:
            return self.cache[name]
        record = self.inventory[name] = {"path": name, "status": "MISSING", "sha256": None}
        if not path.exists():
            require(not required, "required evidence missing: " + name)
            return None
        require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
                "input must be a regular file: " + name)
        before = path.stat()
        record["source_bytes"] = before.st_size
        if before.st_size > self.file_bytes or self.used + before.st_size > self.total_bytes:
            record.update(status="LIMIT", reason="input byte cap; not read or hashed")
            require(not required, "required evidence exceeds byte cap: " + name)
            return None
        with path.open("rb") as handle:
            data = handle.read(self.file_bytes + 1)
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                and len(data) == before.st_size, "source changed while reading: " + name)
        self.used += len(data)
        record.update(status="READ", bytes_read=len(data), sha256=hashlib.sha256(data).hexdigest(),
                      mtime_ns=before.st_mtime_ns)
        if jsonl:
            require(not data or data.endswith(b"\n"), "partial JSONL record: " + name)
            lines = data.splitlines()
            selected = lines[:self.rows]
            value = [json.loads(line) for line in selected if line.strip()]
            require(all(isinstance(r, dict) for r in value), "JSONL row is not an object: " + name)
            checksummed = 0
            for row in value:
                if "record_sha256" in row:
                    require(row["record_sha256"] == digest(
                        {k: v for k, v in row.items() if k != "record_sha256"}
                    ), "JSONL checksum mismatch: " + name)
                    checksummed += 1
            record.update(source_rows=len(lines), parsed_rows=len(selected),
                          truncated=len(lines) > len(selected), checksummed_rows=checksummed,
                          rows_without_checksum=len(value)-checksummed)
        else:
            value = json.loads(data)
        self.cache[name] = value
        return value


def compact(row):
    """Only diagnostic fields; token sequences retain count plus equality digest."""
    result = {"source_row_sha256": digest(row)}
    for key, value in row.items():
        if (key.endswith("token_ids") or key.endswith("tokens")) and isinstance(value, list):
            result[key] = {"count": len(value), "sha256": digest(value)}
        elif key in FIELDS or key.endswith(("_ns", "_ms", "_hash", "_sha256")):
            if key in ("counter_delta", "materialized_positions"):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = compact(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                result[key] = [compact(r) for r in value[:CYCLE_ROWS]]
                if len(value) > CYCLE_ROWS:
                    result[key + "_truncated"] = len(value) - CYCLE_ROWS
            else:
                result[key] = value
    return result


def interval(row):
    for first, last in (("verify_start_ns", "verify_end_ns"), ("start_ns", "end_ns"),
                        ("host_start_ns", "host_end_ns"),
                        ("send_start_ns", "receive_end_ns"),
                        ("received_ns", "completed_ns"),
                        ("work_start_ns", "work_end_ns"),
                        ("start_lower_ns", "end_upper_ns"),
                        ("host_start_ns", "host_launch_end_ns")):
        if type(row.get(first)) is int and type(row.get(last)) is int:
            return row[first], row[last]
    timeline = row.get("timeline")
    if isinstance(timeline, dict):
        return interval(timeline) or (
            (min(times), max(times)) if (times := [v for k, v in timeline.items()
                                                  if k.endswith("_ns") and type(v) is int])
            else None
        )
    return None


def touching(row, start, end):
    span = interval(row)
    return span is not None and span[0] <= end and span[1] >= start


def union(intervals):
    merged = []
    for first, last in sorted((a, b) for a, b in intervals if b > a):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return merged


def intersect(left, right):
    left, right, result, i, j = union(left), union(right), [], 0, 0
    while i < len(left) and j < len(right):
        first, last = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if last > first:
            result.append((first, last))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(rows):
    return sum(b - a for a, b in union(rows)) / 1e6


def bounds(rows, start, end, inner):
    first, last = ("start_upper_ns", "end_lower_ns") if inner else (
        "start_lower_ns", "end_upper_ns")
    fields = ("start_lower_ns", "start_upper_ns", "end_lower_ns", "end_upper_ns")
    require(all(all(type(r.get(k)) is int for k in fields) for r in rows),
            "MISSING native CUDA bounds")
    require(all(0 <= r["start_lower_ns"] <= r["start_upper_ns"]
                and r["end_lower_ns"] <= r["end_upper_ns"]
                and r["start_lower_ns"] < r["end_lower_ns"]
                and r["start_upper_ns"] < r["end_upper_ns"] for r in rows),
            "INVALID native CUDA bound ordering")
    return union((max(start, r[first]), min(end, r[last])) for r in rows)


def device_summary(device, start, end, max_rows):
    raw = device.get("forwards")
    if not isinstance(raw, list):
        return {"status": "MISSING", "reason": "native forwards missing"}
    selected = []
    for index, row in enumerate(raw):
        host = type(row.get("host_start_ns")) is int and start <= row["host_start_ns"] <= end
        physical = (row["start_lower_ns"] <= end and row["end_upper_ns"] >= start
                    if type(row.get("start_lower_ns")) is int
                    and type(row.get("end_upper_ns")) is int else None)
        if host or physical:
            selected.append((index, row, host, physical))
    physical_rows = [r for r in selected if r[3] is True]
    # Preserve physical-window coverage before supplementary host-only rows. A
    # capped host-only tail must not turn a complete physical ZERO into unknown.
    exported = (physical_rows + [r for r in selected if r[3] is not True])[:max_rows]
    exported_indices = {index for index, _, _, _ in exported}
    complete = len(selected) <= max_rows
    physical_status = "COMPLETE" if len(physical_rows) <= max_rows else "PARTIAL"
    try:
        bounds(raw, start, end, False)
    except ValueError as error:
        physical_status = "INVALID" if "INVALID" in str(error) else "MISSING"
    groups = defaultdict(list)
    for info in selected:
        groups[info[1].get("purpose", "target")].append(info)
    purposes = {}
    for purpose, infos in groups.items():
        entries = [r for _, r, _, _ in infos]
        launched = [r for _, r, host, _ in infos if host]
        try:
            lower, upper = bounds(entries, start, end, True), bounds(entries, start, end, False)
            projected = {"clipped_union_lower_ms": duration(lower),
                         "clipped_union_upper_ms": duration(upper)}
        except ValueError as error:
            projected = {"bounds_status": "INVALID" if "INVALID" in str(error) else "MISSING",
                         "reason": str(error)}
        values = [r.get("gpu_event_ms") for r in launched]
        durations_valid = all(type(v) in (int, float) and math.isfinite(v) and v > 0
                              for v in values)
        purposes[purpose] = {
            "window_intersecting_forwards": sum(physical is True for _, _, _, physical in infos),
            "host_launch_selected_forwards": len(launched),
            "launch_selected_B_histogram": dict(Counter(str(r.get("B", "MISSING"))
                                                       for r in launched)),
            "launch_selected_event_sum_ms": sum(values) if durations_valid else None,
            **projected,
        }
    return {"status": "COMPLETE" if complete else "PARTIAL", "source_forward_count": len(raw),
            "window_forward_count": len(physical_rows),
            "host_launch_selected_count": sum(host for _, _, host, _ in selected),
            "physical_window_status": physical_status,
            "host_launch_rows_truncated": any(host and index not in exported_indices
                                              for index, _, host, _ in selected),
            "aggregate_scope": "full source; independent host launches and clipped GPU intervals",
            "truncated": not complete, "identity": device.get("identity", "MISSING"),
            "clock": {k: device.get(k, "MISSING") for k in (
                "clock", "projection_version", "anchor_before_ns", "anchor_after_ns",
                "anchor_uncertainty_ns")}, "by_purpose": purposes,
            "forwards": [{**compact(row), "source_forward_index": index,
                          "host_launch_selected": host, "physical_window_intersects": physical}
                         for index, row, host, physical in exported]}


def project_rows(rows, start, end, limit, source):
    source_status = source.get("status", "MISSING")
    source_truncated = source.get("truncated", False)
    status = source_status if source_status != "READ" else (
        "MISSING" if rows is None else "PARTIAL" if source_truncated else "READ")
    matches = [r for r in rows or [] if touching(r, start, end)]
    projection_truncated = len(matches) > limit
    return {"rows": [compact(r) for r in matches[:limit]],
            "matching_rows": len(matches), "matching_rows_complete": status == "READ",
            "status": status, "source_status": source_status,
            "source_truncated": source_truncated,
            "projection_truncated": projection_truncated,
            "truncated": source_truncated or projection_truncated}


def host_summary(rows, start, end, max_rows):
    if not isinstance(rows, list):
        return {"status": "MISSING", "reason": "host intervals not retained"}
    selected = [r for r in rows if touching(r, start, end)]
    groups = defaultdict(list)
    for row in selected:
        first, last = interval(row)
        groups[row.get("category", "MISSING")].append((max(start, first), min(end, last)))
    return {"status": "COMPLETE", "aggregate_scope": "all source intervals within bounds",
            "by_category": {k: {"count": len(v), "clipped_union_ms": duration(v),
                                "clipped_inclusive_sum_ms": sum(b-a for a, b in v) / 1e6}
                            for k, v in groups.items()},
            "semantics": "nested categories; unions overlap across categories/owners"}


def select_cycles(steps, events, limit):
    chosen = {}

    def add_event(event, reason):
        span = interval(event)
        if span:
            for index, step in enumerate(steps):
                first, last = interval(step)
                if first <= span[1] <= last and len(chosen) < limit:
                    chosen.setdefault(index, []).append(reason)
                    return

    for field in ("promotions", "parent_rejections", "bridge_mismatches"):
        for event in events:
            if event.get("counter_delta", {}).get(field, 0):
                add_event(event, "observed " + field)
                break
    by_request = defaultdict(list)
    for event in sorted(events, key=lambda r: (interval(r) or (0, 0))[1]):
        for rid in event.get("request_ids", ()):
            by_request[rid].append(event)
    trajectory = None
    for rid, entries in by_request.items():
        sequence, cursor = [], 0
        for field in ("promotions", "parent_rejections", "recovery_jobs", "admissions"):
            for index in range(cursor, len(entries)):
                if entries[index].get("counter_delta", {}).get(field, 0):
                    sequence.append(entries[index])
                    cursor = index + 1
                    break
        if len(sequence) == 4:
            trajectory = {"request_id": rid, "events": [compact(r) for r in sequence],
                          "correlation": "request ID and actual owner event order"}
            for event in sequence:
                add_event(event, "same-request promotion/rejection/recovery/eager trajectory")
            break
    for index in (0, len(steps)//2, len(steps)-1):
        if 0 <= index < len(steps) and len(chosen) < limit:
            chosen.setdefault(index, []).append("window position fallback")
    return chosen, trajectory or {"status": "MISSING", "reason": "bounded source lacks trajectory"}


def collect_point(reader, directory, light, cycles):
    start = light["measurement_snapshot"]["measurement_start_ns"]
    end = light["measurement_snapshot"]["measurement_end_ns"]
    require(type(start) is int and type(end) is int and end > start, "invalid measured boundary")
    runtime = reader.read(directory / "runtime.json") or {}
    backend = reader.read(directory / "draft-backend-report.json") or {}

    def source(name):
        return reader.inventory[str((directory / name).relative_to(reader.root))]

    streams, stream_sources, missing = {}, {}, []
    for name in ("runtime.json", "draft-backend-report.json"):
        if source(name)["status"] == "MISSING":
            missing.append(name)
    for name in ("round-events.jsonl", "transport-events.jsonl", "draft-work-events.jsonl",
                 "draft-transport.jsonl"):
        rows = reader.read(directory / name, jsonl=True)
        streams[name], stream_sources[name] = rows, source(name)
        if source(name)["status"] == "MISSING":
            missing.append(name)
    streams["fixed_proposals"] = backend.get("fixed_proposals")
    streams["eager_gpu_steps"] = backend.get("eager_gpu_steps")
    streams["eager_events"] = backend.get("rolling_eager", {}).get("events")
    for name in ("fixed_proposals", "eager_gpu_steps", "eager_events"):
        stream_sources[name] = source("draft-backend-report.json")
    target_devices = runtime.get("target_devices", [])
    devices = {"draft": backend.get("fixed_device") or {}}
    hosts = {"coordinator": runtime.get("host", {}).get("intervals"),
             "draft": backend.get("fixed_host", {}).get("intervals")}
    for index, target in enumerate(target_devices):
        devices[f"target-{index}"] = target.get("device") or {}
        hosts[f"target-{index}"] = target.get("host", {}).get("intervals")
        streams[f"target-{index}-rounds"] = target.get("rounds")
        stream_sources[f"target-{index}-rounds"] = source("runtime.json")
    steps = sorted((s for s in runtime.get("target_steps", [])
                    if s.get("window") and touching(s, start, end)),
                   key=lambda r: (interval(r) or (0, 0))[0])
    events = [r for r in streams["eager_events"] or [] if touching(r, start, end)]
    chosen, trajectory = select_cycles(steps, events, cycles)
    selections = []
    for index, reasons in chosen.items():
        first, last = interval(steps[index])
        selected = {}
        for name, rows in streams.items():
            selected[name] = project_rows(rows, first, last, CYCLE_ROWS, stream_sources[name])
        for name, rows in hosts.items():
            origin = source("draft-backend-report.json" if name == "draft" else "runtime.json")
            selected[name+"-host"] = project_rows(rows, first, last, CYCLE_ROWS, origin)
        eager_starts = [r["host_start_ns"] for r in devices["draft"].get("forwards", [])
                        if r.get("purpose") == "eager"
                        and type(r.get("host_start_ns")) is int
                        and first <= r["host_start_ns"] <= last]
        pre_end = min(eager_starts) if eager_starts else None
        selections.append({"measured_step_index": index, "step": compact(steps[index]),
                           "selection_reasons": reasons, "streams": selected,
                           "host_full_cycle": {k: host_summary(v, first, last, reader.rows)
                                               for k, v in hosts.items()},
                           "before_first_eager_host_launch": {
                               "boundary_ns": [first, pre_end],
                               "semantics": "step start to first recorded eager host launch; "
                                            "not GPU execution boundaries or a causal join",
                               "host": {k: host_summary(v, first, pre_end, reader.rows)
                                        for k, v in hosts.items()} if pre_end is not None
                               else {"status": "MISSING", "reason": "no eager launch in cycle"}},
                           "correlation": "time intersection; IDs retained when source has them"})
    device_reports = {k: device_summary(v, start, end, reader.rows) for k, v in devices.items()}
    eager_rows = [r for r in devices["draft"].get("forwards", [])
                  if r.get("purpose") == "eager"]
    targets = [r for name, d in devices.items() if name != "draft"
               for r in d.get("forwards", [])]
    try:
        require(eager_rows and targets, "native eager or Target forwards missing")
        identities = [d.get("identity", {}) for name, d in devices.items() if name != "draft"]
        require(len(identities) == 2
                and {d.get("global_rank") for d in identities} == {0, 1}
                and all(d.get("gpu_uuid") for d in identities)
                and devices["draft"].get("identity", {}).get("gpu_uuid")
                and len({d.get("gpu_uuid") for d in identities}
                        | {devices["draft"]["identity"]["gpu_uuid"]}) == 3,
                "MISSING or INVALID within-point TP/device identities")
        for report in device_reports.values():
            status = report.get("physical_window_status", "MISSING")
            require(status == "COMPLETE", status + " physical-window records or bounds")
        require(all(type(r.get("gpu_event_ms")) in (int, float)
                    and math.isfinite(r["gpu_event_ms"]) and r["gpu_event_ms"] > 0
                    for r in eager_rows + targets), "INVALID native GPU event duration")
        overlap = {name: duration(intersect(bounds(targets, start, end, inner),
                                            bounds(eager_rows, start, end, inner)))
                   for name, inner in (("lower_ms", True), ("upper_ms", False))}
        overlap["status"] = "RECOMPUTED_FROM_EXPORTED_BOUNDS"
    except ValueError as error:
        overlap = {"status": "INVALID" if "INVALID" in str(error)
                   else "PARTIAL" if "PARTIAL" in str(error) else "MISSING",
                   "reason": str(error)}
    dequeue_sources = [name for name, rows in streams.items()
                       if any("owner_dequeue_ns" in r or "dequeue_ns" in r for r in rows or [])]
    transport_ids = any("request_ids" in r or "request_id" in r
                        for r in streams["transport-events.jsonl"] or [])
    return {
        "source_directory": str(directory), "original_result": light,
        "measurement_boundary_ns": [start, end], "devices": device_reports,
        "target_rank_aggregation": "separate ranks; overlap uses union, never sums rank durations",
        "device_semantics": "model-forward hooks only; launch sums may straddle window edges; "
                            "clipped unions bound wall coverage, not all GPU work or time saved",
        "host": {k: host_summary(v, start, end, reader.rows) for k, v in hosts.items()},
        "recomputed_eager_overlap": overlap, "selected_cycles": selections,
        "same_request_trajectory": trajectory,
        "window_eager_records": {name: project_rows(rows, start, end, reader.rows,
                                                   stream_sources[name])
                                 for name, rows in streams.items()
                                 if name in ("eager_events", "eager_gpu_steps",
                                             "fixed_proposals")},
        "missing_files": missing,
        "limited_files": [Path(r["path"]).name for r in reader.inventory.values()
                          if Path(r["path"]).parent == directory.relative_to(reader.root)
                          and r["status"] == "LIMIT"],
        "field_availability": {
            "owner_dequeue_ns": {"status": "RECORDED" if dequeue_sources else "MISSING",
                                 "sources": dequeue_sources},
            "transport_payload_request_ids": "RECORDED" if transport_ids else "MISSING",
        },
        "missing_is_not_a_new_result_failure": True,
    }


def export(root, output, *, max_file_bytes=MAX_FILE, max_total_bytes=MAX_TOTAL,
           max_jsonl_rows=MAX_ROWS, selected_cycles=MAX_CYCLES):
    require(not Path(output).is_symlink(), "output collision; symlink is not a new path")
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root.is_dir(), "source root does not exist")
    require(not output.is_relative_to(root), "output must be outside source root")
    require(not output.exists(), "output collision; choose a new path")
    require(output.parent.is_dir(), "output parent must already exist")
    require(0 < max_file_bytes <= MAX_FILE and 0 < max_total_bytes <= MAX_TOTAL
            and 0 < max_jsonl_rows <= MAX_ROWS and 0 < selected_cycles <= MAX_CYCLES,
            "limits must be positive and within hard caps")
    reader = Reader(root, max_file_bytes, max_total_bytes, max_jsonl_rows)
    config = reader.read(root / "scan-config.json", required=True)
    require(config.get("execution", {}).get("git_commit") == SOURCE_COMMIT,
            "source scan SHA is not the frozen production commit")
    selected = {}
    for path in sorted((root / "runs").glob("*/point.json")):
        point = reader.read(path, required=True)
        mode = point.get("mode")
        if (mode not in ("serial", "serial-eager") or point.get("batch") != 16
                or point.get("probe")):
            continue
        require(point.get("kind") == "decode-scan" and point.get("scan") is True,
                "only production decode-scan points may be exported")
        require(mode not in selected, "ambiguous duplicate production mode")
        directory = path.parent
        light = reader.read(directory / "light-summary.json", required=True)
        life = reader.read(directory / "process-lifecycle.json", required=True)
        exits = reader.read(directory / "exit-code.json", required=True)
        require(light.get("git_commit") == SOURCE_COMMIT and light.get("mode") == mode
                and light.get("batch") == 16, "production point identity differs")
        require(type(life.get("exit_monotonic_ns")) is int
                and life.get("owned_cleanup_completed") is True
                and life.get("remaining_owned_pids") == []
                and type(exits.get("effective_exit_code")) is int,
                "production point has not ended and completed owned cleanup")
        selected[mode] = (directory, light)
    require(set(selected) == {"serial", "serial-eager"},
            "need exactly both ended B16 production points")
    points = {mode: collect_point(reader, directory, light, selected_cycles)
              for mode, (directory, light) in selected.items()}
    result = {"schema_version": "specrhythm.eager-offline-evidence.v1",
              "source_commit": SOURCE_COMMIT, "source_root": str(root), "points": points,
              "exporter_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "collector_commit_claimed": os.environ.get("SR_EAGER_DIAG_COMMIT"),
              "collector_commit_proof": "environment claim only; caller must verify git HEAD",
              "inventory": list(reader.inventory.values()), "read_bytes": reader.used,
              "limits": {"file_bytes": max_file_bytes, "total_bytes": max_total_bytes,
                         "jsonl_rows": max_jsonl_rows, "selected_cycles": selected_cycles,
                         "records_per_cycle_per_stream": CYCLE_ROWS},
              "read_only_source": True, "inference_started": False,
              "original_result_statuses_unchanged": True}
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    require(len(encoded) <= max_total_bytes, "compact output exceeds total byte cap")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
    return {"output": str(output), "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(), "source_bytes_read": reader.used}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=MAX_FILE)
    parser.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL)
    parser.add_argument("--max-jsonl-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--selected-cycles", type=int, default=MAX_CYCLES)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export(args.root, args.output, max_file_bytes=args.max_file_bytes,
                                max_total_bytes=args.max_total_bytes,
                                max_jsonl_rows=args.max_jsonl_rows,
                                selected_cycles=args.selected_cycles), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, "Offline evidence export: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
