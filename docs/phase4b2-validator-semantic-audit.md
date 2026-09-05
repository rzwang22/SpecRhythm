# Phase 4B.2 offline validator semantic audit

Scope: fix the two false rejection paths introduced by comparison commit
`42fdb4f7f6b38d1dffe6546b011a3823ef192d54`. No GPU execution, execution algorithm changes,
per-mode artifact rewrite, metric-definition change, or new correctness micro-diagnostics.

## Model identity

`configs/phase4b_dual_batch_1d2v.yaml` explicitly sets both model revisions and both tokenizer
revisions to null. `resident_runner.build_decode_ready_context` carries those optional revisions
into the decode-ready manifest. Performance v1 copies target/draft paths and revisions from that
manifest. Local paths with null revisions are legal; truthiness is not an identity validity test.

The repaired validator requires a target and draft model object, a nonempty string path, and a
revision key containing null or a string. Optional revision fields, including tokenizer revisions,
accept null/string and reject malformed types. Entire model dictionaries still compare exactly;
null versus an explicit revision, different paths, missing keys or additional differing identity
fields remain incompatible.

## Authoritative output-length data flow

The following source contracts are present in the original execution commit `56bd0a50...` and
remain unchanged. They do **not** support interpreting the real workload's maximum as a measured
token limit excluding bootstrap.

| Layer | Source | Actual meaning |
|---|---|---|
| Frozen R3-real workload | `phase3/r3_workload.py`, `R3RealRequest.to_dict` | Serializes `maximum_new_tokens`; no `output_tokens` field |
| Request loader | `phase4/stock_vllm.py`, `SmokeRequest.from_dict` | Reads `maximum_new_tokens` directly, requires a positive integer |
| Target runner | `phase4/resident_runner.py`, `SamplingParams` construction | Sets `max_tokens=row.maximum_new_tokens` |
| Serial runner | `phase4/serial_runner.py`, `SamplingParams` construction | Sets `max_tokens=request.maximum_new_tokens` |
| Resident bootstrap | `phase4/resident_vllm.py`, `phase4/vllm_remote.py` | First generated output is the setup bootstrap; it remains in final output |
| Serial output budget | `phase4/vllm_remote.py`, `_generate_initial_resident_proposals` | Initial remaining output budget is `maximum_new_tokens - 1` |
| Serial termination | `phase4/vllm_remote.py`, `_terminal`, `_logical_generated` | Total generated sequence is capped at maximum; EOS can terminate earlier |
| Final output | `phase4/stock_vllm.py`, `_serialize_outputs` | Requires `output.finished`, serializes all `completion.token_ids`, including bootstrap |
| Performance accounting | `phase4/performance.py`, `_request_metrics` | Measured IDs must equal `generated[1:]`; bootstrap is counted separately as one setup token |
| Historical performance metadata | Same function | Incorrectly copies `workload.get("output_tokens")` into `maximum_new_tokens`, yielding null for native R3-real input |

Thus, for source maximum `L`, full-length execution has `len(final)=L`, one setup bootstrap, and
`measured=L-1`. Early stop may have fewer measured tokens. The performance v1 field is not a
reliable copy of `L`: it is null for a native workload. Applying integer checks or arithmetic to
that null rejects every request, independently of the model revision bug. Changing the comparison
to `measured == maximum` would not repair missing metadata and would contradict the executed
source contract.

The old CPU helper used `output_tokens` instead of the native workload key. It supplied a numeric
maximum to the comparator and masked this failure. A new native-workload fixture passes the real
`SmokeRequest.from_dict` loader and the unchanged v1 builder to reproduce the null metadata.

## Completion policy

The comparator consumes per-mode validity, `performance_result`, equal request/completion counts,
positive measured count, final output, terminal evidence and valid token accounting. All cross-mode
request, prompt/bootstrap, count, workload/config, model/vLLM/patch, execution, placement, boundary,
cleanup and overlap gates remain required.

An explicit v1 null maximum is recorded as unavailable legacy metadata. It is not backfilled from
the observed count and does not override the frozen workload SHA. Missing keys, malformed numeric
limits, abort/error/cancel/incomplete reasons, explicit unfinished status and invalid accounting
fail. If a numeric maximum exists, compare it with the **final generated length**, which already
contains bootstrap; reject an overrun or a length finish below that limit. Successful early stops
remain complete. Their different stop labels are diagnostic when measured work matches.

The report's `completion_evidence.null_output_limit_request_counts` discloses missing metadata.
`termination_differences[].fixed_length_completed` is null when the limit is unavailable; this
does not invent evidence of either full-length or early termination. Exact token divergence remains
diagnostic and does not control `performance_valid_for_pair` or the three-mode performance gate.

## Artifact inspection status and reproducible checks

The operator reported individually valid Target/Serial artifacts, 100 completed requests and 1487
measured tokens each, with matching per-request counts and nine exact trajectory differences.
The local workspace contains no real `decode-performance.json` copies. A read-only SSH attempt
through the configured AutoDL970 alias returned connection refused; no app terminal was attached.
Consequently this audit does not claim to have read the real per-request fields or approved the
real pair. The following is the CPU reproduction, **not server measurements**:

| Mode | Workload maximum | v1 artifact maximum | Measured count | Final length | Finish | Stop | Setup count |
|---|---:|---|---:|---:|---|---|---:|
| Target | 3 | null | 2 | 3 | length | null | 1 |
| Serial | 3 | null | 2 | 3 | length | null | 1 |

After binding `SR_PHASE4B2_ROOT` and `SR_PHASE4B_WORKLOAD`, this optional read-only snippet prints
representative existing requests and checks every request against the actual frozen workload.
It reads files only and does not run the measurement builder or any GPU code. The required
continuation in the [runbook](phase4b2-decode-performance-runbook.md#stage-a--reuse-and-approve-the-existing-targetserial-pair-offline)
runs only the two comparison helpers after preserving both old failed v2 files.

```bash
python - "$SR_PHASE4B2_ROOT" "$SR_PHASE4B_WORKLOAD" <<'PY'
import hashlib, json, pathlib, sys
root, workload_path = map(pathlib.Path, sys.argv[1:])
workload = {row["request_id"]: row for row in
            (json.loads(line) for line in workload_path.read_text().splitlines() if line.strip())}
fields = ("maximum_new_tokens", "measured_committed_output_token_count", "finish_reason",
          "termination_reason", "setup_committed_output_tokens")
for mode in ("target", "serial"):
    value = json.loads((root / mode / "decode-performance.json").read_text())
    assert value["valid"] is True and value["performance_result"] is True
    assert hashlib.sha256(workload_path.read_bytes()).hexdigest() == value["workload_sha256"]
    assert value["request_count"] == value["metrics"]["completed_requests"] == len(workload)
    assert {row["request_id"] for row in value["requests"]} == set(workload)
    for index, row in enumerate(value["requests"]):
        frozen = workload[row["request_id"]]
        maximum = frozen["maximum_new_tokens"]
        final = row["total_generated_token_ids"]
        assert row["token_accounting_valid"] is True
        assert row["setup_committed_output_tokens"] == 1
        assert final == [row["bootstrap_token_id"], *row["measured_committed_output_token_ids"]]
        assert row["measured_committed_output_token_count"] == len(final) - 1 > 0
        assert len(final) <= maximum
        if row["finish_reason"] == "length":
            assert len(final) == maximum
        if index < 3 or row["finish_reason"] != "length":
            print(json.dumps({"mode": mode, "request_id": row["request_id"],
                              "workload_maximum_new_tokens": maximum,
                              "workload_output_tokens": frozen.get("output_tokens"),
                              "len(total_generated_token_ids)": len(final),
                              **{key: row.get(key) for key in fields}}, sort_keys=True))
PY
```

The CPU regression suite covers the native legacy-null field shape, nullable model identity,
positive measured work with bootstrap counted exactly once, successful early stop, failed and
contradictory completion evidence, all existing strict comparability gates, and a synthetic
100-request / 1487-token pair with nine diagnostic-only differences. The synthetic pair preserves
the supplied aggregate metric values and never substitutes for server artifact verification.
