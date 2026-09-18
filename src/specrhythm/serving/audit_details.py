"""Lossless post-run report tables, bounded shards; never change measurement evidence."""

import copy
import hashlib
import json
from pathlib import Path

from specrhythm.serving.audit_layer_report import MAX_OUTPUT, write
from specrhythm.serving.common import require

SCHEMA = "specrhythm.audit-detail-tables.v1"
MAX_SHARDS = 8
TABLES = (
    ("pingpong", "cycles"),
    ("pingpong", "pipeline", "dispatch", "rows"),
    ("pingpong", "pipeline", "timeline"),
    ("pingpong", "draft_dispatch", "calls"),
)


def container(value, path):
    for key in path[:-1]:
        value = value[key]
    return value


def publish(value, output):
    """All validation precedes projection; summary is published after its detail files.

    Each shard retains the original row and table order. At most eight existing
    8 MiB-sized units (64 MiB total), no growing limit or silent truncation. These
    are derived joins, not copies of runtime/backend or new hot-path records.
    """
    output = Path(output)
    summary = copy.deepcopy(value)
    if not value.get("mode", "").endswith("-k3"):
        return write(summary, output)
    tables = []
    for path in TABLES:
        try:
            parent = container(summary, path)
        except KeyError:
            continue  # An absent input remains absent, never an empty evidence table.
        rows = parent.get(path[-1])
        if isinstance(rows, list) and rows:
            tables.append((path, parent.pop(path[-1])))
    binding = {k: value.get(k) for k in (
        "source_commit", "mode", "workload_sha256", "measurement_start_ns", "measurement_end_ns"
    )}
    files, payload, count = [], bytearray(), 0

    def flush():
        nonlocal payload, count
        if not payload:
            return
        require(len(files) < MAX_SHARDS, "audit details exceed bounded shard count",
                maximum_shards=MAX_SHARDS, maximum_shard_bytes=MAX_OUTPUT)
        path = output.with_name(output.stem + f"-details.{len(files):03d}.jsonl")
        with path.open("xb") as handle:
            handle.write(payload)
        files.append(dict(name=path.name, bytes=len(payload), rows=count,
                          sha256=hashlib.sha256(payload).hexdigest()))
        payload, count = bytearray(), 0

    for table, rows in tables:
        for index, row in enumerate(rows):
            encoded = (json.dumps(dict(table=list(table), index=index, row=row),
                                  separators=(",", ":"), sort_keys=True) + "\n").encode()
            require(len(encoded) <= MAX_OUTPUT, "single audit detail row exceeds shard bound",
                    table=list(table), index=index, actual_bytes=len(encoded),
                    maximum_bytes=MAX_OUTPUT)
            if len(payload) + len(encoded) > MAX_OUTPUT:
                flush()
            payload.extend(encoded)
            count += 1
    flush()
    summary["detail_tables"] = dict(
        schema_version=SCHEMA, binding=binding,
        tables=[dict(path=list(p), rows=len(rows)) for p, rows in tables], files=files,
        maximum_shards=MAX_SHARDS, maximum_shard_bytes=MAX_OUTPUT,
        projection="LOSSLESS_EXTERNAL_TABLES; summary statistics and missing values unchanged",
    )
    receipt = write(summary, output)
    raw = output.read_bytes()
    require(len(raw) == receipt["bytes"]
            and hashlib.sha256(raw).hexdigest() == receipt["sha256"],
            "audit summary readback differs", file=str(output))
    restore_file(json.loads(raw), output.parent)
    return receipt


def restore(summary, read_blob):
    """Verify bytes/count/order/binding before exposing externalized original rows.

    read_blob receives a validated basename and may read the run directory or a
    digest-addressed archive object. Missing/corrupt/oversized evidence is an error.
    Historical inline reports retain their original schema and behavior.
    """
    if "detail_tables" not in summary:
        return summary
    spec = summary["detail_tables"]
    require(spec.get("schema_version") == SCHEMA, "unknown audit detail schema")
    require(spec.get("maximum_shards") == MAX_SHARDS
            and spec.get("maximum_shard_bytes") == MAX_OUTPUT, "audit detail limits differ")
    require(spec.get("binding") == {k: summary.get(k) for k in (
        "source_commit", "mode", "workload_sha256", "measurement_start_ns", "measurement_end_ns"
    )}, "audit detail report binding differs")
    result, tables, names = copy.deepcopy(summary), {}, set()
    for item in spec["tables"]:
        path, count = tuple(item["path"]), item["rows"]
        require(path in TABLES and path not in tables and type(count) is int and count > 0,
                "invalid/duplicate audit detail table")
        parent = container(result, path)
        require(path[-1] not in parent, "audit detail table conflicts with inline rows")
        parent[path[-1]] = []
        tables[path] = count
    require(len(spec["files"]) <= MAX_SHARDS, "audit detail shard count exceeds bound")
    for entry in spec["files"]:
        name = entry["name"]
        require(isinstance(name, str) and Path(name).name == name and name not in names
                and "-audit-report-details." in name and name.endswith(".jsonl"),
                "invalid/duplicate audit detail filename")
        names.add(name)
        require(type(entry["bytes"]) is int and 0 < entry["bytes"] <= MAX_OUTPUT,
                "invalid audit detail byte count")
        raw = read_blob(name)
        require(len(raw) == entry["bytes"]
                and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                "audit detail size/SHA256 mismatch", file=name)
        require(raw.endswith(b"\n"), "incomplete audit detail framing", file=name)
        count = 0
        for line in raw.splitlines():
            row = json.loads(line)
            path = tuple(row["table"])
            require(path in tables, "unexpected audit detail table", file=name)
            rows = container(result, path)[path[-1]]
            require(type(row["index"]) is int and row["index"] == len(rows)
                    and len(rows) < tables[path],
                    "audit detail row order/count differs", file=name)
            rows.append(row["row"])
            count += 1
        require(type(entry["rows"]) is int and count == entry["rows"],
                "audit detail shard row count differs", file=name)
    for path, count in tables.items():
        require(len(container(result, path)[path[-1]]) == count, "audit detail table incomplete")
    del result["detail_tables"]
    return result


def restore_file(summary, directory):
    def read(name):
        with (Path(directory) / name).open("rb") as handle:
            return handle.read(MAX_OUTPUT + 1)

    return restore(summary, read)
