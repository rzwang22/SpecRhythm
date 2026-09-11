"""Read-only, allowlisted fixed timing inputs; no runtime or GPU imports."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

FILES = {
    "light-summary.json",
    "runtime.json",
    "draft-backend-report.json",
    "draft-work-events.jsonl",
    "draft-transport.jsonl",
}
MODES = ("serial", "pingpong", "serial-split", "target")
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_EXPORT_BYTES = 5 * 1024 * 1024  # Reserve remaining budget for reports/progress/manifest.


def keep(row, fields):
    return {k: row[k] for k in fields.split() if k in row}


def atomic_json(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("attribution report exceeds 4 MiB; raw traces must not enter reports")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def mode_from_path(name):
    for part in PurePosixPath(name).parts:
        for mode in sorted(MODES, key=len, reverse=True):
            if part.startswith("continuous-" + mode + "-"):
                return mode
    return None


class Inputs:
    def __init__(self, source, modes, progress):
        self.source, self.progress = Path(source), progress
        self.archive = tarfile.open(source, "r:*") if self.source.is_file() else None
        self.entries = {}
        if self.archive:
            # Inspect headers only; never extract paths or follow links.
            for number, member in enumerate(self.archive):
                if number >= 20000:
                    raise ValueError("input archive exceeds 20000-member inspection budget")
                if member.isfile():
                    if member.name in self.entries:
                        raise ValueError("duplicate archive member: " + member.name)
                    self.entries[member.name] = member
        else:
            runs = self.source / "runs"
            directories = sorted(runs.iterdir()) if runs.is_dir() else [self.source]
            for directory in directories:
                if not directory.is_dir() or directory.is_symlink():
                    continue
                for filename in sorted(FILES):
                    for suffix in ("", ".gz"):
                        path = directory / (filename + suffix)
                        if path.is_file() and not path.is_symlink():
                            name = path.relative_to(self.source).as_posix()
                            # A standalone attempt keeps its actual attempt identity.
                            if directory == self.source:
                                name = self.source.name + "/" + path.name
                            self.entries[name] = path
        self.names = []
        for name in self.entries:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe input member: " + name)
            base = path.name.removesuffix(".gz")
            if base in FILES and mode_from_path(name) in modes:
                self.names.append(name)
        self.names.sort()
        if len(self.names) > 160:
            raise ValueError("select at most 32 attempts (160 allowlisted artifacts)")
        self.digests = []

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, name):
        self.progress(current_file=name, phase="reading")
        entry = self.entries[name]
        raw_stream = self.archive.extractfile(entry) if self.archive else entry.open("rb")
        stream = raw_stream
        try:
            if name.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=stream)
            digest = hashlib.sha256()
            data = bytearray()
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                data.extend(chunk)
                digest.update(chunk)
                if len(data) > MAX_INPUT_BYTES:
                    raise ValueError("input exceeds 512 MiB per-file budget: " + name)
                self.progress(bytes_read=len(chunk))
            self.digests.append(
                {
                    "artifact": name,
                    "uncompressed_bytes": len(data),
                    "uncompressed_sha256": digest.hexdigest(),
                }
            )
            if name.removesuffix(".gz").endswith(".jsonl"):
                rows = []
                for line, raw in enumerate(io.BytesIO(data), 1):
                    if not raw.endswith(b"\n"):
                        raise ValueError(f"incomplete event record: {name}:{line}")
                    row = project_event(json.loads(raw))
                    row["source_line"] = line
                    rows.append(row)
                    self.progress(events=1)
                value = rows
            else:
                value = json.loads(data)
                self.progress(events=count_records(value))
            self.progress(processed_file=name, phase="parsed")
            return value
        finally:
            stream.close()
            raw_stream.close()


def count_records(value):
    if isinstance(value, list):
        return sum(1 + count_records(r) for r in value if isinstance(r, dict))
    if isinstance(value, dict):
        return sum(count_records(v) for v in value.values())
    return 0


def project_event(row):
    result = keep(
        row,
        "schema_version event event_id operation request_id request_ids "
        "proposal_id round_id prefix_version logical_cohort cohort start_ns end_ns "
        "timestamp_ns success error transport serialization request_payload_bytes "
        "response_payload_bytes blocks_on_draft_gpu measurement_region source_line",
    )
    if "record_sha256" in row:
        # This is the ORIGINAL record hash, not a checksum of the redacted projection.
        result["source_record_sha256"] = row["record_sha256"]
    elif "source_record_sha256" in row:
        result["source_record_sha256"] = row["source_record_sha256"]
    if isinstance(row.get("result"), dict):
        r = row["result"]
        result["result"] = keep(
            r,
            "request_id proposal_id round_id prefix_version "
            "draft_sync_complete_ns target_tail_ready_ns terminal target_tail "
            "logical_cohort logical_draft_kv_length",
        )
        if isinstance(r.get("proposal"), dict):
            result["result"]["proposal"] = keep(
                r["proposal"],
                "request_id proposal_id round_id prefix_version "
                "prefix_token_count parent_prefix_len prefix_token_sha256 "
                "draft_start_ns draft_end_ns created_timestamp_ns "
                "proposal_token_ids logical_cohort",
            )
    return result


LIGHT_FIELDS = (
    "schema_version mode point valid errors execution_status measurement_status cleanup_status "
    "execution_sha256 effective_exit_code window_ms window_throughput_tok_s actual_rotation_ms "
    "committed_window_tokens g_per_request_per_verification actual_target_batch "
    "actual_draft_forward_batch draft_forward_count target_forward_count "
    "pipeline_stage_gpu_event_ms overlap host_observation warmup_ms drain_ms "
    "arrival_to_drain_ms startup_and_state_preparation_ms stop_reason valid_samples "
    "cross_run_token_equality cross_run_length_equality cross_run_EOS_equality "
    "cross_run_round_equality uuid_query_by_rank recorded_gpu_costs diagnostic_logging"
)


def project(name, value):
    """Only needed timing/identity/short commit evidence, never prefixes or KV maps."""
    name = PurePosixPath(name).name.removesuffix(".gz")
    if name.endswith(".jsonl"):
        return [project_event(r) for r in value]
    if name == "light-summary.json":
        return keep(value, LIGHT_FIELDS)
    if name == "runtime.json":
        result = keep(
            value,
            "schema_version point measurement_start_ns measurement_end_ns "
            "warmup_start_ns warmup_end_ns warmup_steps sample_count stop_reason "
            "start_ns end_ns target_steps host prompt_lengths events population",
        )
        result["requests"] = [
            keep(
                r,
                "request_id cohort state admission_ns completion_ns "
                "release_ns resources_released commits",
            )
            for r in value.get("requests", [])
        ]
        result["target_devices"] = [
            keep(d, "device host rounds target_rows") for d in value.get("target_devices", [])
        ]
        return result
    result = keep(
        value,
        "schema_version fixed_proposals fixed_device fixed_host "
        "draft_model_forward_count_by_purpose",
    )
    result["s2_work_records"] = [
        keep(
            r, "operation logical_cohort request_ids host_start_ns host_end_ns terminal_by_request"
        )
        for r in value.get("s2_work_records", [])
    ]
    return result


def export_file(output, name, value, budget):
    """Bound total compressed export; publish each complete artifact atomically."""
    path = output / "evidence" / (name.removesuffix(".gz") + ".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=1) as zipped:
                encoder = json.JSONEncoder(separators=(",", ":"), allow_nan=False)
                rows = value if name.removesuffix(".gz").endswith(".jsonl") else [value]
                chunks, buffered = [], 0
                for row in rows:
                    for chunk in encoder.iterencode(row):
                        chunks.append(chunk)
                        buffered += len(chunk)
                        if buffered >= 65536:
                            zipped.write("".join(chunks).encode())
                            chunks, buffered = [], 0
                            if raw.tell() + budget[0] > MAX_EXPORT_BYTES:
                                raise ValueError(
                                    "5 MiB evidence budget exceeded; select fewer attempts"
                                )
                    chunks.append("\n")
                zipped.write("".join(chunks).encode())
        budget[0] += temporary.stat().st_size
        if budget[0] > MAX_EXPORT_BYTES:
            raise ValueError("5 MiB evidence budget exceeded; select fewer attempts")
        temporary.replace(path)
        return {
            "artifact": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    finally:
        temporary.unlink(missing_ok=True)
